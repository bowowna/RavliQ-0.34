# QwenAnalog — Аналог архитектуры Qwen 2.5 (2.5B)

Полная реализация архитектуры с нуля, без предобученных весов.
Ты сам подбираешь датасеты и обучаешь модель.

---

## 📁 Структура файлов

```
qwen_analog/
├── model.py          ← Вся архитектура модели
├── train.py          ← Скрипт обучения
├── train_config.json ← Конфигурация (параметры)
└── README.md
```

---

## 🏗️ Архитектура (аналог Qwen 2.5)

| Компонент | Значение |
|-----------|---------|
| Параметры | ~2.5B |
| Слои | 36 |
| Hidden size | 2048 |
| Attention heads | 16 |
| KV heads (GQA) | 8 |
| FFN size | 11008 |
| Макс. контекст | 32768 токенов |
| Позиционные энкодинги | RoPE (theta=1M) |
| Нормализация | RMSNorm |
| Активация FFN | SwiGLU |
| Attention | Grouped Query Attention (GQA) |

---

## 🚀 Быстрый старт

### 1. Установка зависимостей

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install transformers datasets
```

### 2. Подготовь датасет

Формат — JSONL файл, каждая строка:
```json
{"text": "Твой текст здесь..."}
```

### 3. Запуск обучения

```bash
# Одна GPU
python train.py --data ./my_dataset.jsonl --config train_config.json

# Продолжение обучения с чекпоинта
python train.py --data ./my_dataset.jsonl --resume ./checkpoints/step_1000/checkpoint.pt
```

### 4. Multi-GPU (DDP)

```bash
torchrun --nproc_per_node=4 train.py --data ./my_dataset.jsonl
```

---

## 🔧 Настройка токенизатора

В `train.py` замени токенизатор на свой:

```python
# Вариант 1: Qwen токенизатор (BPE, 151936 токенов)
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B")

# Вариант 2: SentencePiece свой
import sentencepiece as spm
sp = spm.SentencePieceProcessor()
sp.Load("my_tokenizer.model")

# Вариант 3: tiktoken
import tiktoken
enc = tiktoken.get_encoding("cl100k_base")
```

Не забудь обновить `vocab_size` в `train_config.json`!

---

## 💾 Требования к железу

| Режим | VRAM |
|-------|------|
| Inference (fp32) | ~10GB |
| Inference (bf16) | ~5GB |
| Обучение (batch=1, grad_ckpt) | ~24GB |
| Обучение (batch=4) | ~80GB |

**Рекомендуется:** A100 80GB или несколько RTX 3090/4090.

---

## 📊 Использование модели

```python
import torch
from model import QwenForCausalLM, QwenConfig
from transformers import AutoTokenizer

# Инициализация
config = QwenConfig()
model = QwenForCausalLM(config)

# Загрузка весов после обучения
ckpt = torch.load("checkpoints/final/checkpoint.pt")
model.load_state_dict(ckpt["model_state_dict"])
model.eval()

# Токенизатор
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B")

# Генерация
prompt = "Привет, как дела?"
input_ids = torch.tensor([tokenizer.encode(prompt)])
output = model.generate(input_ids, max_new_tokens=100, temperature=0.7)
print(tokenizer.decode(output[0]))

# Подсчёт параметров
print(f"Параметров: {model.count_parameters()/1e9:.2f}B")
```

---

## 📈 Советы по обучению

1. **Разогрев LR** — добавь warmup первые 1-2% шагов
2. **Gradient checkpointing** — включи при нехватке VRAM
3. **bf16** — быстрее fp16 на современных GPU
4. **Batch size** — используй gradient accumulation для имитации большого батча
5. **Датасет** — минимум 1-10B токенов для нормального обучения LLM

---

## 🔄 Отличия от оригинального Qwen 2.5

- Нет интеграции с HuggingFace `transformers` (но легко добавить)
- Упрощённая генерация (без beam search)
- Нет Flash Attention (добавь `flash-attn` для ускорения)

Для добавления Flash Attention:
```python
# pip install flash-attn
from flash_attn import flash_attn_func
# Замени стандартный softmax attention в GQAttention.forward()
```
