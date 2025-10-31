from transformers import (
    TrainingArguments,
    Trainer,
    AutoTokenizer,
    AutoModelForCausalLM,
    DataCollatorForLanguageModeling,
)
from peft import get_peft_model, LoraConfig, TaskType
from datasets import load_dataset
import torch
import os
from transformers import BitsAndBytesConfig

# ✅ CUDA fragmentation mitigation
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# ✅ Model & tokenizer
model_id = "NousResearch/Nous-Hermes-2-Mistral-7B-DPO"
max_memory = {0: "4GB", "cpu": "15GB"}
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,  # atau torch.bfloat16 kalau kamu pakai A100/H100
    bnb_4bit_quant_type='nf4',  # ✅ penting: 'nf4' bisa jalan di CPU, 'fp4' tidak bisa
)

tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    device_map="cpu",
    quantization_config=bnb_config,
    # load_in_4bit=True,
    offload_folder="offload",
    max_memory=max_memory,
    torch_dtype=torch.float16,   
    # llm_int8_enable_fp32_cpu_offload=True,
)

# ✅ Gradient checkpointing + LoRA + require grad
model.gradient_checkpointing_enable()
model = get_peft_model(
    model,
    LoraConfig(
        r=8,
        lora_alpha=32,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    ),
)
model.enable_input_require_grads()

# ✅ Dataset
dataset = load_dataset("json", data_files="app/faq_tune.json")["train"]

def tokenize(example):
    prompt = f"### Pertanyaan:\n{example['input']}\n\n### Jawaban:\n{example['output']}"
    return tokenizer(prompt, truncation=True, padding="max_length", max_length=512)

tokenized_dataset = dataset.map(tokenize, remove_columns=dataset.column_names)

# ✅ TrainingArguments
training_args = TrainingArguments(
    per_device_train_batch_size=1,
    gradient_accumulation_steps=4,
    warmup_steps=10,
    num_train_epochs=3,
    learning_rate=2e-4,
    fp16=True,
    logging_steps=10,
    output_dir="./roxy-ai",
    save_strategy="epoch",
)

# ✅ Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    tokenizer=tokenizer,
    data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
)

# ✅ Clear cache and train
# torch.cuda.empty_cache()
trainer.train()
# torch.cuda.empty_cache()
