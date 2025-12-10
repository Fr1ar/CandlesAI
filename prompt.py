from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

base = "Qwen/Qwen2.5-7B-Instruct"
lora = "./lora_level_adapter"

tokenizer = AutoTokenizer.from_pretrained(base)
model = AutoModelForCausalLM.from_pretrained(base, device_map="auto")

model = PeftModel.from_pretrained(model, lora)

prompt = "Сделай очень сложный уровень где ключ закрыт красными свечами"
x = tokenizer(prompt, return_tensors="pt").to(model.device)

y = model.generate(**x, max_new_tokens=200)
print(tokenizer.decode(y[0], skip_special_tokens=True))
