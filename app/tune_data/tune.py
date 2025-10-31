# gagal
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from peft import get_peft_model, LoraConfig, TaskType
from trl import SFTTrainer
from datasets import load_dataset
from transformers import BitsAndBytesConfig

# Load base model & tokenizer
model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
bnb_config = BitsAndBytesConfig(
    load_in_8bit=True,
    llm_int8_enable_fp32_cpu_offload=True  # penting ini
)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="cpu"
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
    gradient_accumulation_steps=4,
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

trainer.train()
