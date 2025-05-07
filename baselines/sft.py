import unsloth
import argparse
from trl import SFTTrainer
from transformers import TrainingArguments
from unsloth import FastLanguageModel
from dataset_prep import dataset
import torch

max_seq_length = 2048
dtype = None 
load_in_4bit = False 

# Argument parser setup
parser = argparse.ArgumentParser(description="Train LLaMA with Unsloth and LoRA")
parser.add_argument(
    "--model_name",
    type=str,
    default="unsloth/llama-2-7b",
    help="The name or path of the pretrained model to load"
)
args = parser.parse_args()

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = args.model_name, # You can change this to any Llama model!
    max_seq_length = max_seq_length,
    dtype = dtype,
    load_in_4bit = load_in_4bit,
)

model = FastLanguageModel.get_peft_model(
    model,
    r = 16,
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                      "gate_proj", "up_proj", "down_proj",],
    lora_alpha = 16,
    lora_dropout = 0, # Currently only supports dropout = 0
    bias = "none",    # Currently only supports bias = "none"
    use_gradient_checkpointing = True,
    random_state = 3407,
    max_seq_length = max_seq_length,
)

trainer = SFTTrainer(
    model = model,
    train_dataset = dataset["train"],
    eval_dataset=dataset["test"],
    dataset_text_field = "text",
    max_seq_length = max_seq_length,
    args = TrainingArguments(
        per_device_train_batch_size = 16,
        gradient_accumulation_steps = 4,
        warmup_steps = 5,
        num_train_epochs=1,
        learning_rate = 2e-4,
        fp16 = not torch.cuda.is_bf16_supported(),
        bf16 = torch.cuda.is_bf16_supported(),
        logging_steps = 1,
        optim = "adamw_8bit",
        weight_decay = 0.01,
        lr_scheduler_type = "linear",
        seed = 3407,
        output_dir = "outputs",
        report_to="wandb",
    ),
)

trainer_stats = trainer.train()
trainer_eval = trainer.evaluate()