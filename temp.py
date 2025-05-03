# train.py

import pandas as pd
import torch
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)

print("🚀 Начинаем процесс дообучения модели...")

# --- Пути ---
DATASET_PATH = "src/data/faq_full.parquet"
OUTPUT_DIR = "src/models/finetuned_rugpt"

# --- Загрузка данных ---
df = pd.read_parquet(DATASET_PATH)  # можно заменить на pd.read_json(...) если нужно


# Форматирование примеров
def format_example(row):
    return f"Вопрос: {row['user_question']}\nОтвет: {row['assistant_answer']}</s>"


df["text"] = df.apply(format_example, axis=1)
dataset = Dataset.from_pandas(df[["text"]])

# --- Модель и токенизатор ---
MODEL_NAME = "ai-forever/rugpt3small_based_on_gpt2"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)


# --- Токенизация ---
def tokenize_function(examples):
    return tokenizer(
        examples["text"], truncation=True, max_length=512, padding="max_length"
    )


tokenized_datasets = dataset.map(
    tokenize_function, batched=True, num_proc=4, remove_columns=["text"]
)

# --- Data collator ---
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False,
)

# --- Настройки обучения ---
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    overwrite_output_dir=True,
    num_train_epochs=3,
    per_device_train_batch_size=2,
    save_steps=10_000,
    save_total_limit=2,
    logging_dir="logs",
    logging_steps=100,
    fp16=torch.cuda.is_available(),
    learning_rate=2e-5,
    warmup_steps=500,
    weight_decay=0.01,
    report_to="none",
    push_to_hub=False,
)

trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=tokenized_datasets,
    tokenizer=tokenizer,
)

# --- Обучение ---
print("🧠 Начинаю обучение...")
trainer.train()

# --- Сохранение модели ---
print(f"💾 Сохраняю модель в {OUTPUT_DIR}...")
trainer.save_model(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)

print("✅ Дообучение завершено!")
