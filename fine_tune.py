import os
import logging
import json
from datasets import Dataset, load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

DATA_FILE = os.path.join(os.path.dirname(__file__), "..", "data", "prompt_injections.json")

def load_data(data_file: str):
    """Loads the prompt injection dataset from a JSON file."""
    with open(data_file, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data

def format_example(example: dict) -> str:
    """
    Format a training example by concatenating the input prompt and the safe response.
    You can adjust the delimiter or add special tokens as needed.
    """
    return example["input"].strip() + "\nResponse: " + example["target"].strip()

def prepare_dataset(data):
    """
    Convert a list of examples into a Hugging Face Dataset.
    """
    formatted_texts = [format_example(example) for example in data]
    return Dataset.from_dict({"text": formatted_texts})

def tokenize_function(examples, tokenizer):
    return tokenizer(
        examples["text"],
        padding="max_length",
        truncation=True,
        max_length=128,  # adjust max_length based on your requirements
    )

def main():
    logger.info("Loading dataset from %s", DATA_FILE)
    data = load_data(DATA_FILE)
    dataset = prepare_dataset(data)

    # need pretrained model here though, fine tune on deepseek bc open source. If you want, change to LLaMa, but idk why you would do that
    model_name = "deepseek-r1"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(model_name)

    logger.info("Tokenizing dataset...")
    tokenized_dataset = dataset.map(lambda x: tokenize_function(x, tokenizer), batched=True, remove_columns=["text"])

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)


    training_args = TrainingArguments(
        output_dir="./fine_tuned_model",
        overwrite_output_dir=True,
        num_train_epochs=3,
        per_device_train_batch_size=2,
        save_steps=10,
        save_total_limit=2,
        logging_steps=5,
        evaluation_strategy="no",
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )

    logger.info("Starting fine-tuning process to mitigate prompt injections...")
    trainer.train()
    trainer.save_model()
    logger.info("Model fine-tuning complete. The fine-tuned model is saved to './fine_tuned_model'.")

if __name__ == "__main__":
    main()
