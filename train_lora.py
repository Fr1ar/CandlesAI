from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
import torch

MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct"  # можно заменить на LLaMA
DATA_PATH = "levels.jsonl"  # твой датасет

# 1. Загружаем данные
dataset = load_dataset("json", data_files=DATA_PATH)


# Объединяем prompt + target в единый текст
def format_sample(example):
    example["text"] = f"""### Пользователь:
{example["prompt"]}

### Уровень:
{example["level"]}
"""
    return example


dataset = dataset.map(format_sample)

# 2. Загружаем токенизатор и модель
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token

# Optional: quantization (сильно экономит VRAM)
device_map = "auto"
load_in_8bit = False
load_in_4bit = True

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    device_map=device_map,
    load_in_4bit=load_in_4bit,
    torch_dtype=torch.bfloat16,
    trust_remote_code=True,
)

# 3. Настраиваем LoRA
lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    lora_dropout=0.1,
    bias="none",
    target_modules=["q_proj", "v_proj"],  # для Qwen и LLaMA
)

model = prepare_model_for_kbit_training(model)
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()


# 4. Токенизация
def tokenize(example):
    return tokenizer(
        example["text"], truncation=True, padding="max_length", max_length=512
    )


tokenized = dataset.map(tokenize, batched=True)

# 5. TrainingArguments
args = TrainingArguments(
    output_dir="./lora-level-generator",
    per_device_train_batch_size=1,
    gradient_accumulation_steps=8,
    warmup_steps=50,
    max_steps=1000,  # хватит для первого обучения
    learning_rate=2e-4,
    fp16=True,
    logging_steps=10,
    save_steps=200,
)

# 6. Trainer
trainer = Trainer(
    model=model,
    args=args,
    train_dataset=tokenized["train"],
)

# 7. Запуск обучения
trainer.train()

# 8. Сохраняем LoRA адаптер
model.save_pretrained("./lora_level_adapter")
print("Готово! LoRA-сети сохранены.")
