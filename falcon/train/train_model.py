import sys
sys.path.insert(1,'/home/nair.ro/LitArt/falcon')


import transformers
from transformers import Trainer, TrainingArguments, DataCollatorForLanguageModeling
from models.base_model import get_base_model
from data.tokenization import tokenize_input
from data.dataset import load_tokenized_dataset
from utils.logger import get_logger
from utils.parameters import (batch_size, epochs, gradient_accumulation_steps, learning_rate, 
                                     save_total_limit, logging_steps, output_dir, max_steps, log_path)

def train_model():
    model, tokenizer = get_base_model()
    tokenized_dataset = load_tokenized_dataset(tokenize_input, tokenizer, 1024, 128)

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
        optim="paged_adamw_8bit",
        lr_scheduler_type='cosine',
        warmup_ratio=0.05,
    )

    trainer = Trainer(
        model=model,
        train_dataset=tokenized_dataset,
        args=training_args,
        data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
        callbacks=[get_logger(log_path)]
    )

    trainer.train()
    model.save_pretrained(log_path)
    trainer.save_model(output_dir)
