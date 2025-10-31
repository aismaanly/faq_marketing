# gagal
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from peft import get_peft_model, LoraConfig, TaskType
from trl import SFTTrainer
from datasets import load_dataset
from transformers import BitsAndBytesConfig
import torch
# Load base model & tokenizer
model_name = "NousResearch/Nous-Hermes-2-Mistral-7B-DPO"
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4"
)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="auto",
    quantization_config=bnb_config,
    trust_remote_code=True
)
tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)

# LoRA config
lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=["q_proj", "v_proj"],  # tergantung model
    lora_dropout=0.1,
    bias="none",
    task_type=TaskType.CAUSAL_LM,
)

model = get_peft_model(model, lora_config)

# Load dataset custom atau pakai dummy
def format_qa(example):
    return {
        "text": f"### Pertanyaan:\n{example['input']}\n\n### Jawaban:\n{example['output']}"
    }

dataset = load_dataset("json", data_files="app/faq_tune.json")
dataset = dataset["train"].map(format_qa)

# Training args
training_args = TrainingArguments(
    per_device_train_batch_size=1,
    gradient_accumulation_steps=1,
    num_train_epochs=3,
    learning_rate=2e-4,
    logging_steps=10,
    output_dir="./lora-mistral",
    save_total_limit=2,
    fp16=True
)

trainer = SFTTrainer(
    model=model,
    train_dataset=dataset,
    args=training_args,
    # dataset_text_field="text"  # WAJIB jika kolomnya bukan default
)
# trainer = SFTTrainer(
#     model=model,
#     train_dataset=dataset,
#     # tokenizer=tokenizer,
#     args=training_args,
# )

torch.cuda.empty_cache()
torch.cuda.ipc_collect()
trainer.train()
