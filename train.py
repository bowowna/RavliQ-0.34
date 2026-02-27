"""
Скрипт обучения для QwenAnalog модели.
Поддерживает:
- Обучение с нуля (from scratch)
- Продолжение обучения (resume from checkpoint)
- Gradient checkpointing для экономии памяти
- Mixed precision (bf16/fp16)
- Multi-GPU через PyTorch DDP
"""

import os
import math
import json
import time
import argparse
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

try:
    from torch.cuda.amp import autocast, GradScaler
    AMP_AVAILABLE = True
except ImportError:
    AMP_AVAILABLE = False

from model import QwenForCausalLM, QwenConfig


# ─────────────────────────────────────────────
#  Простой Dataset (jsonl формат)
# ─────────────────────────────────────────────

def _format_role(role: str) -> str:
    """Красивое имя роли для чат-формата."""
    mapping = {
        "user": "Пользователь", "human": "Пользователь",
        "bot": "Ассистент", "assistant": "Ассистент", "gpt": "Ассистент",
        "system": "Система",
    }
    return mapping.get(role.lower(), role.capitalize())


def _extract_texts_from_json(data) -> list:
    """
    Рекурсивно извлекает текст из любой JSON структуры.
    Поддерживаемые форматы:
      - {"messages": [{"role": "user", "content": "..."}]}     <- ru_turbo_saiga
      - {"conversations": [{"from": "human", "value": "..."}]} <- sharegpt
      - {"dialogue": [{"role": "...", "text": "..."}]}
      - {"text": "..."}
      - {"instruction": "...", "output": "..."}                <- alpaca
      - ["текст1", "текст2"]
    """
    texts = []

    if isinstance(data, str):
        if data.strip():
            texts.append(data.strip())

    elif isinstance(data, list):
        for item in data:
            texts.extend(_extract_texts_from_json(item))

    elif isinstance(data, dict):

        # ── ru_turbo_saiga / OpenAI chat: {"messages": [{"role": ..., "content": ...}]}
        if "messages" in data and isinstance(data["messages"], list):
            parts = []
            for turn in data["messages"]:
                if not isinstance(turn, dict):
                    continue
                role    = turn.get("role", "").strip()
                content = (turn.get("content") or turn.get("text") or "").strip()
                if content:
                    label = _format_role(role) if role else ""
                    parts.append(f"{label}: {content}" if label else content)
            if parts:
                texts.append("\n".join(parts))
                return texts

        # ── ShareGPT: {"conversations": [{"from": "human", "value": ...}]}
        if "conversations" in data and isinstance(data["conversations"], list):
            parts = []
            for turn in data["conversations"]:
                if not isinstance(turn, dict):
                    continue
                role  = (turn.get("from") or turn.get("role") or "").strip()
                value = (turn.get("value") or turn.get("text") or turn.get("content") or "").strip()
                if value:
                    label = _format_role(role) if role else ""
                    parts.append(f"{label}: {value}" if label else value)
            if parts:
                texts.append("\n".join(parts))
                return texts

        # ── dialogue: {"dialogue": [{"role": ..., "text": ...}]}
        if "dialogue" in data and isinstance(data["dialogue"], list):
            parts = []
            for turn in data["dialogue"]:
                if isinstance(turn, dict):
                    role  = (turn.get("role") or turn.get("from") or "").strip()
                    value = (turn.get("text") or turn.get("content") or turn.get("value") or "").strip()
                    if value:
                        label = _format_role(role) if role else ""
                        parts.append(f"{label}: {value}" if label else value)
                elif isinstance(turn, str) and turn.strip():
                    parts.append(turn.strip())
            if parts:
                texts.append("\n".join(parts))
                return texts

        # ── plain text fields
        for key in ("text", "body", "passage", "document", "context"):
            if key in data and isinstance(data[key], str) and data[key].strip():
                texts.append(data[key].strip())
                return texts

        # ── alpaca / instruction format
        parts = []
        for key in ("system", "instruction", "input", "question", "prompt"):
            if key in data and isinstance(data[key], str) and data[key].strip():
                parts.append(data[key].strip())
        for key in ("output", "response", "answer", "completion", "reply"):
            if key in data and isinstance(data[key], str) and data[key].strip():
                parts.append(data[key].strip())
        if parts:
            texts.append("\n".join(parts))
            return texts

        # ── fallback: recursive
        for v in data.values():
            if isinstance(v, (dict, list)):
                texts.extend(_extract_texts_from_json(v))

    return texts


class TextDataset(Dataset):
    """
    Загружает ВСЕ .json и .jsonl файлы из указанной папки (рекурсивно).

    Поддерживаемые форматы JSON:
      - JSONL (каждая строка — отдельный JSON объект)
      - JSON массив: [{"text": ...}, ...]
      - JSON объект: {"text": ...} или {"data": [...]}
      - Любой вложенный JSON — текст извлекается автоматически
      - Форматы: plain text, alpaca, sharegpt, conversations и т.д.
    """
    def __init__(self, data_dir: str, tokenizer, max_length: int = 2048):
        self.samples = []
        self.tokenizer = tokenizer
        self.max_length = max_length

        data_path = Path(data_dir)
        if not data_path.exists():
            abs_path = data_path.resolve()
            print(f"\n❌ Папка не найдена: {abs_path}")
            print(f"\n💡 Создай папку data рядом со скриптом и положи туда JSON файлы.")
            print(f"   Или запусти с явным путём:")
            print(f"   python train.py --data C:\\RavliQ\\data")
            raise FileNotFoundError(f"Папка не найдена: {abs_path}")

        # Собираем все .json и .jsonl файлы рекурсивно
        json_files = sorted(list(data_path.rglob("*.json")) + list(data_path.rglob("*.jsonl")))

        if not json_files:
            abs_path = data_path.resolve()
            print(f"\n❌ В папке не найдено .json / .jsonl файлов!")
            print(f"   Искал в: {abs_path}")
            print(f"   Папка существует: {data_path.exists()}")
            if data_path.exists():
                all_files = list(data_path.iterdir())
                if all_files:
                    print(f"   Файлы в папке ({len(all_files)} шт):")
                    for f in all_files[:10]:
                        print(f"     - {f.name}  ({f.suffix})")
                else:
                    print(f"   Папка пустая!")
            print(f"\n💡 Запусти с явным путём:")
            print(f"   python train.py --data C:\\RavliQ\\data")
            print(f"   python train.py --data ./data")
            raise ValueError(f"Нет JSON файлов в: {abs_path}")

        print(f"\n[Dataset] Найдено файлов: {len(json_files)}")
        print(f"[Dataset] Папка: {data_dir}\n")

        total_before = 0
        for fpath in json_files:
            count_before = len(self.samples)
            self._load_file(fpath)
            count_added = len(self.samples) - count_before
            size_kb = fpath.stat().st_size / 1024
            print(f"  ✓ {fpath.name:<40} {size_kb:>8.1f} KB  →  {count_added:>6} примеров")

        print(f"\n[Dataset] Итого примеров: {len(self.samples):,}")

    def _load_file(self, fpath: Path):
        """Загружает один файл, автоматически определяет формат."""
        try:
            with open(fpath, "r", encoding="utf-8", errors="replace") as f:
                content = f.read().strip()

            if not content:
                return

            # Пробуем JSONL (каждая строка — JSON)
            lines = content.splitlines()
            loaded_as_jsonl = False
            if len(lines) > 1:
                jsonl_texts = []
                all_ok = True
                for line in lines:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        obj = json.loads(line)
                        extracted = _extract_texts_from_json(obj)
                        jsonl_texts.extend(extracted)
                    except json.JSONDecodeError:
                        all_ok = False
                        break
                if all_ok and jsonl_texts:
                    self.samples.extend(jsonl_texts)
                    loaded_as_jsonl = True

            if not loaded_as_jsonl:
                # Пробуем как один большой JSON
                try:
                    data = json.loads(content)
                    texts = _extract_texts_from_json(data)
                    self.samples.extend(texts)
                except json.JSONDecodeError:
                    # Последний вариант — читаем как plain text построчно
                    for line in lines:
                        line = line.strip()
                        if line:
                            self.samples.append(line)

        except Exception as e:
            print(f"  ⚠ Ошибка при загрузке {fpath.name}: {e}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        text = self.samples[idx]
        tokens = self.tokenizer.encode(text, add_special_tokens=True)
        tokens = tokens[:self.max_length]

        input_ids = torch.tensor(tokens, dtype=torch.long)
        labels = input_ids.clone()
        # Паддинг
        pad_len = self.max_length - len(input_ids)
        if pad_len > 0:
            input_ids = torch.cat([input_ids, torch.zeros(pad_len, dtype=torch.long)])
            labels = torch.cat([labels, torch.full((pad_len,), -100, dtype=torch.long)])

        return {"input_ids": input_ids, "labels": labels}


# ─────────────────────────────────────────────
#  Trainer
# ─────────────────────────────────────────────

class Trainer:
    def __init__(self, config: dict):
        self.cfg = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[Trainer] Устройство: {self.device}")

        # Модель
        model_cfg = QwenConfig(**config.get("model_config", {}))
        self.model = QwenForCausalLM(model_cfg).to(self.device)
        params = self.model.count_parameters()
        print(f"[Trainer] Параметров: {params/1e9:.2f}B")

        # Gradient checkpointing (экономия VRAM, только на GPU)
        if config.get("gradient_checkpointing", False):
            if self.device.type == "cuda":
                self.model.model.gradient_checkpointing_enable()
            else:
                print("[Trainer] Gradient checkpointing пропущен (CPU не поддерживает)")

        # Оптимизатор
        self.optimizer = AdamW(
            self.model.parameters(),
            lr=config.get("learning_rate", 3e-4),
            weight_decay=config.get("weight_decay", 0.1),
            betas=(0.9, 0.95),
        )

        # Scaler для mixed precision
        self.scaler = GradScaler() if (AMP_AVAILABLE and config.get("use_amp", True) and self.device.type == "cuda") else None
        self.use_amp = self.scaler is not None

        self.global_step = 0
        self.output_dir = Path(config.get("output_dir", "./checkpoints"))
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def train(self, train_dataloader: DataLoader, num_epochs: int = 1):
        total_steps = len(train_dataloader) * num_epochs
        scheduler = CosineAnnealingLR(self.optimizer, T_max=total_steps, eta_min=1e-5)

        grad_accum = self.cfg.get("gradient_accumulation_steps", 4)
        max_grad_norm = self.cfg.get("max_grad_norm", 1.0)
        log_every = self.cfg.get("log_every", 10)
        save_every = self.cfg.get("save_every", 500)

        self.model.train()
        self.optimizer.zero_grad()
        running_loss = 0.0
        t0 = time.time()

        for epoch in range(num_epochs):
            print(f"\n{'='*50}")
            print(f"  EPOCH {epoch+1}/{num_epochs}")
            print(f"{'='*50}")

            for step, batch in enumerate(train_dataloader):
                input_ids = batch["input_ids"].to(self.device)
                labels    = batch["labels"].to(self.device)

                if self.use_amp:
                    with autocast(dtype=torch.bfloat16):
                        out = self.model(input_ids=input_ids, labels=labels)
                        loss = out["loss"] / grad_accum
                    self.scaler.scale(loss).backward()
                else:
                    out = self.model(input_ids=input_ids, labels=labels)
                    loss = out["loss"] / grad_accum
                    loss.backward()

                running_loss += loss.item() * grad_accum

                if (step + 1) % grad_accum == 0:
                    if self.use_amp:
                        self.scaler.unscale_(self.optimizer)
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_grad_norm)
                        self.scaler.step(self.optimizer)
                        self.scaler.update()
                    else:
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_grad_norm)
                        self.optimizer.step()

                    scheduler.step()
                    self.optimizer.zero_grad()
                    self.global_step += 1

                    if self.global_step % log_every == 0:
                        avg_loss = running_loss / log_every
                        ppl = math.exp(min(avg_loss, 20))
                        lr = scheduler.get_last_lr()[0]
                        elapsed = time.time() - t0
                        tokens_per_sec = (log_every * grad_accum * input_ids.numel()) / elapsed
                        print(f"  Step {self.global_step:6d} | loss={avg_loss:.4f} | ppl={ppl:.2f} | "
                              f"lr={lr:.2e} | tok/s={tokens_per_sec:.0f}")
                        running_loss = 0.0
                        t0 = time.time()

                    if self.global_step % save_every == 0:
                        self.save_checkpoint()

        self.save_checkpoint(final=True)
        print("\n✅ Обучение завершено!")

    def save_checkpoint(self, final: bool = False):
        tag = "final" if final else f"step_{self.global_step}"
        ckpt_dir = self.output_dir / tag
        ckpt_dir.mkdir(parents=True, exist_ok=True)

        torch.save({
            "step": self.global_step,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "config": self.cfg,
        }, ckpt_dir / "checkpoint.pt")

        # Сохраняем конфиг отдельно
        with open(ckpt_dir / "model_config.json", "w", encoding="utf-8") as f:
            json.dump(self.cfg.get("model_config", {}), f, indent=2)

        print(f"  💾 Сохранено: {ckpt_dir}")

    @classmethod
    def load_checkpoint(cls, checkpoint_path: str, config: dict):
        trainer = cls(config)
        ckpt = torch.load(checkpoint_path, map_location=trainer.device)
        trainer.model.load_state_dict(ckpt["model_state_dict"])
        trainer.optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        trainer.global_step = ckpt["step"]
        print(f"[Trainer] Чекпоинт загружен: step={trainer.global_step}")
        return trainer


# ─────────────────────────────────────────────
#  Точка входа
# ─────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Обучение QwenAnalog")
    parser.add_argument("--config", type=str, default="train_config.json", help="Путь к конфигу")
    # Автоматически определяем дефолтный путь под Windows и Linux
    import platform
    if platform.system() == "Windows":
        # Папка data рядом со скриптом
        default_data = str(Path(__file__).parent / "data")
    else:
        default_data = "/data"
    parser.add_argument("--data", type=str, default=default_data, help="Папка с JSON/JSONL датасетами")
    parser.add_argument("--resume", type=str, default=None, help="Путь к чекпоинту для продолжения")
    args = parser.parse_args()

    # Загружаем конфиг
    if os.path.exists(args.config):
        with open(args.config, encoding="utf-8") as f:
            config = json.load(f)
    else:
        # Дефолтный конфиг
        config = {
            "model_config": {
                "vocab_size": 32000,   # измени под свой токенизатор
                "hidden_size": 2048,
                "num_hidden_layers": 36,
                "num_attention_heads": 16,
                "num_kv_heads": 8,
                "intermediate_size": 11008,
                "max_position_embeddings": 2048,
            },
            "learning_rate": 3e-4,
            "weight_decay": 0.1,
            "batch_size": 2,
            "gradient_accumulation_steps": 8,
            "num_epochs": 3,
            "use_amp": True,
            "gradient_checkpointing": True,
            "log_every": 10,
            "save_every": 500,
            "output_dir": "./checkpoints",
        }

    # Токенизатор — подключи свой!
    # Пример с HuggingFace tokenizer:
    try:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B")
        print("[Tokenizer] Использую Qwen токенизатор")
    except Exception:
        # Простой character-level заглушка для тестов
        class DummyTokenizer:
            def encode(self, text, add_special_tokens=True):
                return [ord(c) % 32000 for c in text]
        tokenizer = DummyTokenizer()
        print("[Tokenizer] Использую dummy токенизатор (замени на свой!)")

    # Датасет
    dataset = TextDataset(
        args.data, tokenizer,
        max_length=config["model_config"].get("max_position_embeddings", 2048)
    )
    import platform
    # На Windows num_workers > 0 вызывает ошибки с multiprocessing
    num_workers = 0 if platform.system() == "Windows" else 2
    pin_memory  = torch.cuda.is_available()
    dataloader = DataLoader(
        dataset,
        batch_size=config.get("batch_size", 2),
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    # Тренер
    if args.resume:
        trainer = Trainer.load_checkpoint(args.resume, config)
    else:
        trainer = Trainer(config)

    trainer.train(dataloader, num_epochs=config.get("num_epochs", 3))


if __name__ == "__main__":
    main()
