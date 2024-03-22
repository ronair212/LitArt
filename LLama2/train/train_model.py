import sys
sys.path.insert(1,'/home/nair.ro/test/LitArt/falcon')


import transformers
from trl import SFTTrainer
from transformers import Trainer, TrainingArguments, DataCollatorForLanguageModeling

from Llama2.models.base_model import get_base_model
from Llama2.data.tokenization import tokenize_input
from Llama2.data.dataset import load_dataset_FN
from Llama2.configs.bnb_config import get_bnb_config
from Llama2.configs.lora_config import get_lora_config
from Llama2.utils.logger import get_logger , save_hyperparameters 
from Llama2.utils.parameters import (batch_size, epochs, gradient_accumulation_steps, learning_rate, 
                                     save_total_limit, logging_steps, output_dir, max_steps, log_path)

def train_model():
    base_model, tokenizer = get_base_model()
    data = load_dataset_FN()
    

    training_args = TrainingArguments(
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        num_train_epochs=epochs,
        learning_rate=learning_rate,
        fp16=True,
        save_total_limit=save_total_limit,
        logging_steps=logging_steps,
        output_dir=output_dir,
        max_steps=max_steps,
        save_strategy='epoch',
        optim="adamw_hf",  #"paged_adamw_32bit" #"paged_adamw_8bit"
        lr_scheduler_type='cosine',
        warmup_ratio=0.05,
        max_grad_norm=0.3,
        gradient_checkpointing=True,
    )
    

    def formatting_func(example):
        text = f"### USER: Summarize the following text : {example['chapter']}\n### ASSISTANT: {example['summary_text']}"
        return text



    trainer = SFTTrainer(
                        model=base_model,
                        args=training_args,
                        train_dataset=data["train"],
                        packing=True,
                        tokenizer=tokenizer,
                        max_seq_length=1024,
                        formatting_func=formatting_func,
                        )

    
            
        
    save_hyperparameters(log_path, bnb_config , lora_config , training_arguments)



    trainer.train()
    model.save_pretrained(output_dir)
    trainer.save_model(output_dir)
    
