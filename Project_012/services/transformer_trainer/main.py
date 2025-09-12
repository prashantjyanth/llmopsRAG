import torch
from transformers import (
    AutoTokenizer,
    PegasusConfig,
    PegasusTokenizer,
    PegasusForConditionalGeneration,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments
)
from datasets import load_dataset
import evaluate
import numpy as np

# Load samsum dataset
dataset = load_dataset("knkarthick/samsum")
train_dataset = dataset["train"]
eval_dataset = dataset["validation"]

# Tokenizer and model
model_checkpoint = "google/pegasus-xsum"
tokenizer = PegasusTokenizer.from_pretrained(model_checkpoint)

light_config = PegasusConfig(
    d_model=256,
    encoder_layers=3,
    decoder_layers=3,
    encoder_attention_heads=4,
    decoder_attention_heads=4,
    decoder_ffn_dim=1024,
    encoder_ffn_dim=1024,
    vocab_size=tokenizer.vocab_size,
    max_position_embeddings=512,
)
model = PegasusForConditionalGeneration(light_config)

# Preprocess
max_input_length = 512
max_target_length = 128

def preprocess_function(examples):
    inputs = examples["dialogue"]
    targets = examples["summary"]
    model_inputs = tokenizer(
        inputs, max_length=max_input_length, truncation=True, padding="max_length"
    )
    labels = tokenizer(
        targets, max_length=max_target_length, truncation=True, padding="max_length"
    )
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

tokenized_train = train_dataset.map(
    preprocess_function, batched=True, remove_columns=train_dataset.column_names
)
tokenized_eval = eval_dataset.map(
    preprocess_function, batched=True, remove_columns=eval_dataset.column_names
)

# Metrics
rouge = evaluate.load("rouge")
bertscore = evaluate.load("bertscore")

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    # ROUGE
    rouge_result = rouge.compute(predictions=decoded_preds, references=decoded_labels)
    # BERTScore
    bertscore_result = bertscore.compute(
        predictions=decoded_preds, references=decoded_labels, lang="en"
    )

    return {
        **rouge_result,
        "bertscore_precision": np.mean(bertscore_result["precision"]),
        "bertscore_recall": np.mean(bertscore_result["recall"]),
        "bertscore_f1": np.mean(bertscore_result["f1"]),
    }

# Training
training_args = Seq2SeqTrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    weight_decay=0.01,
    save_total_limit=2,
    num_train_epochs=1,
    predict_with_generate=True,
    logging_dir="./logs",
    logging_steps=10,
)

trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train,
    eval_dataset=tokenized_eval,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
)

trainer.train()
metrics = trainer.evaluate()
print("Training complete.")
print("Evaluation metrics:")
for k, v in metrics.items():
    print(f"{k}: {v}")
