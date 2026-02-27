"""
Аналог архитектуры Qwen 2.5 (2.5B параметров)
Чистая архитектура без весов — готова для обучения на своих датасетах.

Архитектурные особенности (как у Qwen 2.5):
- GQA (Grouped Query Attention)
- RoPE (Rotary Position Embeddings)
- RMSNorm вместо LayerNorm
- SwiGLU активация в FFN
- Sliding Window Attention (опционально)
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
from dataclasses import dataclass


# ─────────────────────────────────────────────
#  Конфигурация модели
# ─────────────────────────────────────────────

@dataclass
class QwenConfig:
    vocab_size: int = 151936       # как у оригинального Qwen
    hidden_size: int = 2048        # размер скрытого слоя
    num_hidden_layers: int = 36    # количество слоёв трансформера
    num_attention_heads: int = 16  # кол-во голов внимания
    num_key_value_heads: int = 8   # кол-во KV голов (GQA)
    intermediate_size: int = 11008 # размер FFN
    max_position_embeddings: int = 32768  # макс. длина контекста
    rope_theta: float = 1000000.0  # RoPE theta (как у Qwen2.5)
    rms_norm_eps: float = 1e-6
    initializer_range: float = 0.02
    use_sliding_window: bool = False
    sliding_window: int = 4096
    tie_word_embeddings: bool = False
    hidden_act: str = "silu"
    pad_token_id: int = 0
    bos_token_id: int = 1
    eos_token_id: int = 2
    dropout: float = 0.0


# ─────────────────────────────────────────────
#  RMSNorm
# ─────────────────────────────────────────────

class RMSNorm(nn.Module):
    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        variance = x.pow(2).mean(-1, keepdim=True)
        x = x * torch.rsqrt(variance + self.eps)
        return self.weight * x


# ─────────────────────────────────────────────
#  Rotary Position Embeddings (RoPE)
# ─────────────────────────────────────────────

class RotaryEmbedding(nn.Module):
    def __init__(self, dim: int, max_seq_len: int = 32768, theta: float = 1000000.0):
        super().__init__()
        self.dim = dim
        self.max_seq_len = max_seq_len
        self.theta = theta
        self._build_cache(max_seq_len)

    def _build_cache(self, seq_len: int):
        freqs = 1.0 / (self.theta ** (torch.arange(0, self.dim, 2).float() / self.dim))
        t = torch.arange(seq_len).float()
        freqs = torch.outer(t, freqs)
        emb = torch.cat([freqs, freqs], dim=-1)
        self.register_buffer("cos_cached", emb.cos()[None, None, :, :], persistent=False)
        self.register_buffer("sin_cached", emb.sin()[None, None, :, :], persistent=False)

    def forward(self, x: torch.Tensor, seq_len: int):
        if seq_len > self.max_seq_len:
            self._build_cache(seq_len)
        return (
            self.cos_cached[:, :, :seq_len, :].to(x.device),
            self.sin_cached[:, :, :seq_len, :].to(x.device),
        )


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat([-x2, x1], dim=-1)


def apply_rotary_emb(q, k, cos, sin):
    q_rot = (q * cos) + (rotate_half(q) * sin)
    k_rot = (k * cos) + (rotate_half(k) * sin)
    return q_rot, k_rot


# ─────────────────────────────────────────────
#  Grouped Query Attention (GQA)
# ─────────────────────────────────────────────

class GQAttention(nn.Module):
    def __init__(self, config: QwenConfig, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx

        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.num_kv_groups = self.num_heads // self.num_kv_heads

        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=True)
        self.k_proj = nn.Linear(self.hidden_size, self.num_kv_heads * self.head_dim, bias=True)
        self.v_proj = nn.Linear(self.hidden_size, self.num_kv_heads * self.head_dim, bias=True)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=False)

        self.rotary_emb = RotaryEmbedding(
            dim=self.head_dim,
            max_seq_len=config.max_position_embeddings,
            theta=config.rope_theta,
        )

    def _repeat_kv(self, x: torch.Tensor) -> torch.Tensor:
        """Повторяем KV головы для GQA"""
        bs, num_kv_heads, seq_len, head_dim = x.shape
        if self.num_kv_groups == 1:
            return x
        x = x[:, :, None, :, :].expand(bs, num_kv_heads, self.num_kv_groups, seq_len, head_dim)
        return x.reshape(bs, num_kv_heads * self.num_kv_groups, seq_len, head_dim)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        bs, seq_len, _ = hidden_states.size()

        q = self.q_proj(hidden_states).view(bs, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(hidden_states).view(bs, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(hidden_states).view(bs, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)

        cos, sin = self.rotary_emb(q, seq_len=seq_len)
        q, k = apply_rotary_emb(q, k, cos, sin)

        if past_key_value is not None:
            k = torch.cat([past_key_value[0], k], dim=2)
            v = torch.cat([past_key_value[1], v], dim=2)

        past_kv = (k, v) if use_cache else None

        k = self._repeat_kv(k)
        v = self._repeat_kv(v)

        scale = math.sqrt(self.head_dim)
        attn_weights = torch.matmul(q, k.transpose(2, 3)) / scale

        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask

        attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(q.dtype)

        attn_output = torch.matmul(attn_weights, v)
        attn_output = attn_output.transpose(1, 2).contiguous().view(bs, seq_len, self.hidden_size)
        attn_output = self.o_proj(attn_output)

        return attn_output, past_kv


# ─────────────────────────────────────────────
#  SwiGLU FFN (как у Qwen 2.5)
# ─────────────────────────────────────────────

class QwenMLP(nn.Module):
    def __init__(self, config: QwenConfig):
        super().__init__()
        self.gate_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.up_proj   = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.down_proj = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)
        self.act_fn    = nn.SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # SwiGLU: gate * SiLU(x) ⊙ up(x)
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))


# ─────────────────────────────────────────────
#  Decoder Layer
# ─────────────────────────────────────────────

class QwenDecoderLayer(nn.Module):
    def __init__(self, config: QwenConfig, layer_idx: int):
        super().__init__()
        self.self_attn = GQAttention(config, layer_idx)
        self.mlp = QwenMLP(config)
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple] = None,
        use_cache: bool = False,
    ):
        # Pre-norm Attention
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states, past_kv = self.self_attn(
            hidden_states, attention_mask, position_ids, past_key_value, use_cache
        )
        hidden_states = residual + hidden_states

        # Pre-norm FFN
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        return hidden_states, past_kv


# ─────────────────────────────────────────────
#  Основная модель
# ─────────────────────────────────────────────

class QwenModel(nn.Module):
    """Базовая модель без LM головы"""
    def __init__(self, config: QwenConfig):
        super().__init__()
        self.config = config
        self.gradient_checkpointing = False
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id)
        self.layers = nn.ModuleList([
            QwenDecoderLayer(config, i) for i in range(config.num_hidden_layers)
        ])
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def gradient_checkpointing_enable(self):
        """Включает gradient checkpointing — экономит VRAM за счёт пересчёта активаций."""
        self.gradient_checkpointing = True
        print("[Model] Gradient checkpointing ВКЛЮЧЁН (экономия памяти)")

    def gradient_checkpointing_disable(self):
        self.gradient_checkpointing = False

    def _make_causal_mask(self, seq_len: int, dtype: torch.dtype, device: torch.device):
        mask = torch.full((seq_len, seq_len), float("-inf"), device=device, dtype=dtype)
        mask = torch.triu(mask, diagonal=1)
        return mask[None, None, :, :]

    def forward(
        self,
        input_ids: torch.LongTensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[list] = None,
        use_cache: bool = False,
    ):
        bs, seq_len = input_ids.shape
        hidden_states = self.embed_tokens(input_ids)

        causal_mask = self._make_causal_mask(seq_len, hidden_states.dtype, hidden_states.device)

        if past_key_values is None:
            past_key_values = [None] * len(self.layers)

        new_past = []
        for i, layer in enumerate(self.layers):
            if self.gradient_checkpointing and self.training:
                # gradient checkpointing: пересчитываем активации вместо хранения
                import torch.utils.checkpoint as cp
                def create_custom_forward(l):
                    def custom_forward(*inputs):
                        out, _ = l(inputs[0], inputs[1], None, None, False)
                        return out
                    return custom_forward
                hidden_states = cp.checkpoint(
                    create_custom_forward(layer),
                    hidden_states, causal_mask,
                    use_reentrant=False
                )
                past_kv = None
            else:
                hidden_states, past_kv = layer(
                    hidden_states,
                    attention_mask=causal_mask,
                    position_ids=position_ids,
                    past_key_value=past_key_values[i],
                    use_cache=use_cache,
                )
            new_past.append(past_kv)

        hidden_states = self.norm(hidden_states)
        return hidden_states, new_past if use_cache else None


class QwenForCausalLM(nn.Module):
    """Модель с LM головой для авторегрессионного обучения"""
    def __init__(self, config: QwenConfig):
        super().__init__()
        self.config = config
        self.model = QwenModel(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        if config.tie_word_embeddings:
            self.lm_head.weight = self.model.embed_tokens.weight

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=self.config.initializer_range)

    def forward(
        self,
        input_ids: torch.LongTensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        past_key_values: Optional[list] = None,
        use_cache: bool = False,
    ):
        hidden_states, past_kv = self.model(
            input_ids, attention_mask, position_ids, past_key_values, use_cache
        )
        logits = self.lm_head(hidden_states)

        loss = None
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss = F.cross_entropy(
                shift_logits.view(-1, self.config.vocab_size),
                shift_labels.view(-1),
                ignore_index=-100,
            )

        return {"loss": loss, "logits": logits, "past_key_values": past_kv}

    def generate(
        self,
        input_ids: torch.LongTensor,
        max_new_tokens: int = 256,
        temperature: float = 0.7,
        top_p: float = 0.9,
        top_k: int = 50,
        repetition_penalty: float = 1.1,
        eos_token_id: Optional[int] = None,
    ) -> torch.LongTensor:
        """Простая генерация с sampling"""
        self.eval()
        eos = eos_token_id or self.config.eos_token_id
        generated = input_ids.clone()
        past = None

        with torch.no_grad():
            for _ in range(max_new_tokens):
                out = self.forward(generated if past is None else generated[:, -1:],
                                   past_key_values=past, use_cache=True)
                past = out["past_key_values"]
                logits = out["logits"][:, -1, :].float()

                # Repetition penalty
                for prev_id in generated[0].tolist():
                    logits[0, prev_id] /= repetition_penalty

                # Temperature
                logits = logits / max(temperature, 1e-8)

                # Top-K
                if top_k > 0:
                    top_k_vals, _ = torch.topk(logits, top_k)
                    logits[logits < top_k_vals[:, -1:]] = float("-inf")

                # Top-P (nucleus)
                probs = F.softmax(logits, dim=-1)
                sorted_probs, sorted_idx = torch.sort(probs, descending=True)
                cumsum = torch.cumsum(sorted_probs, dim=-1)
                remove = cumsum - sorted_probs > top_p
                sorted_probs[remove] = 0.0
                sorted_probs /= sorted_probs.sum()
                next_token = sorted_idx.gather(-1, torch.multinomial(sorted_probs, 1))

                generated = torch.cat([generated, next_token], dim=-1)
                if next_token.item() == eos:
                    break

        return generated

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())
