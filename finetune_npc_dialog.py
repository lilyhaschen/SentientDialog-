from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments, TextDataset, DataCollatorForLanguageModeling
import os

# Settings
model_name = "microsoft/DialoGPT-medium"
data_path = "npc_dialog_data.txt"  # Your conversation samples here
output_dir = "npc_finetuned_model"

# Load model and tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# Load dataset
def load_dataset(file_path, tokenizer, block_size=128):
    return TextDataset(
        tokenizer=tokenizer,
        file_path=file_path,
        block_size=block_size
    )

# Prepare training components
dataset = load_dataset(data_path, tokenizer)
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

# Training arguments
training_args = TrainingArguments(
    output_dir=output_dir,
    overwrite_output_dir=True,
    num_train_epochs=3,
    per_device_train_batch_size=2,
    save_steps=500,
    save_total_limit=2,
    logging_steps=100,
    prediction_loss_only=True,
)

# Trainer setup
trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=dataset,
)

# Train
trainer.train()

# Save model
trainer.save_model(output_dir)
tokenizer.save_pretrained(output_dir)
print(f"Model fine-tuned and saved to '{output_dir}'")

'''
finetune_npc_dialog.py: Fine-tuning Script for Custom NPC Personality
--------------------------------------------------------
This script fine-tunes DialoGPT-medium using conversational data that reflects a specific NPC's tone or backstory.
Use for offline fine-tuning of character-based dialogue models.
'''
