import pandas as pd
import json
import torch
import random
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from transformers import (
    BertTokenizer,
    BertForSequenceClassification,
    Trainer,
    TrainingArguments,
    EarlyStoppingCallback
)
from torch.utils.data import Dataset

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
set_seed(42)

df = pd.read_excel("fine_tuning.xlsx", engine="openpyxl")

with open("label_map.json", "r", encoding="utf-8") as f:
    label_map = json.load(f)

label2id = {k: v for k, v in label_map.items()}
id2label = {v: k for k, v in label_map.items()}
classes = list(label2id.keys())
num_labels = len(classes)
df["label_id"] = df["reason"].map(label2id)

train_texts, val_texts, train_labels, val_labels = train_test_split(
    df["text"], df["label_id"],
    test_size=0.2, random_state=42, stratify=df["label_id"]
)

tokenizer = BertTokenizer.from_pretrained("./bert_reason_classifier")
model = BertForSequenceClassification.from_pretrained(
    "./bert_reason_classifier",
    num_labels=num_labels
)

class FineTuneDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len=128):
        self.encodings = tokenizer(
            texts.tolist(),
            truncation=True,
            padding="max_length",
            max_length=max_len,
            return_tensors="pt"
        )
        self.labels = torch.tensor(labels.tolist(), dtype=torch.long)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {key: val[idx] for key, val in self.encodings.items()}
        item["labels"] = self.labels[idx]
        return item

train_dataset = FineTuneDataset(train_texts, train_labels, tokenizer)
val_dataset = FineTuneDataset(val_texts, val_labels, tokenizer)

def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    acc = accuracy_score(labels, preds)
    f1 = f1_score(labels, preds, average="macro")
    precision = precision_score(labels, preds, average="macro", zero_division=0)
    recall = recall_score(labels, preds, average="macro", zero_division=0)
    return {
        "accuracy": acc,
        "macro_f1": f1,
        "precision": precision,
        "recall": recall
    }

training_args = TrainingArguments(
    output_dir="./fine_tune_results",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    learning_rate=1e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=10,
    weight_decay=0.01,
    logging_dir="./fine_tune_logs",
    logging_steps=10,
    save_total_limit=2,
    metric_for_best_model="macro_f1",
    greater_is_better=True,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    compute_metrics=compute_metrics,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=2)]
)

if __name__ == "__main__":
    trainer.train()
    model.save_pretrained("./bert_reason_classifier_finetuned")
    tokenizer.save_pretrained("./bert_reason_classifier_finetuned")
