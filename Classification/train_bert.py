import pandas as pd
import numpy as np
import random
import json
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
import torch
from torch.utils.data import Dataset
from transformers import (
    BertTokenizer,
    BertForSequenceClassification,
    Trainer,
    TrainingArguments,
    EarlyStoppingCallback,
    TrainerCallback,
)

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
set_seed(42)

df = pd.read_csv("weibo_negative_with_one_reason.csv", encoding="utf-8-sig")
df = df[df["reason"] != "其他"].copy()

le = LabelEncoder()
df["label_id"] = le.fit_transform(df["reason"])
num_labels = len(le.classes_)

label_map = {k: int(v) for k, v in zip(le.classes_, le.transform(le.classes_))}
with open("label_map.json", "w", encoding="utf-8") as f:
    json.dump(label_map, f, ensure_ascii=False, indent=4)

train_texts, val_texts, train_labels, val_labels = train_test_split(
    df["text1"], df["label_id"],
    test_size=0.2, random_state=42, stratify=df["label_id"]
)
train_df = pd.DataFrame({'text1': train_texts, 'label_id': train_labels})
val_df = pd.DataFrame({'text1': val_texts, 'label_id': val_labels})
train_df.to_csv("train_split.csv", index=False, encoding="utf-8-sig")
val_df.to_csv("val_split.csv", index=False, encoding="utf-8-sig")

tokenizer = BertTokenizer.from_pretrained("bert-base-chinese")

class WeiboDataset(Dataset):
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

train_dataset = WeiboDataset(train_texts, train_labels, tokenizer)
val_dataset = WeiboDataset(val_texts, val_labels, tokenizer)

model = BertForSequenceClassification.from_pretrained(
    "bert-base-chinese",
    num_labels=num_labels
)

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
        "macro_precision": precision,
        "macro_recall": recall
    }

training_args = TrainingArguments(
    output_dir="./results",
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=15,
    weight_decay=0.01,
    logging_dir="./logs",
    logging_steps=100,
    save_total_limit=2,
    metric_for_best_model="macro_f1",
    greater_is_better=True,
)

class LoggingCallback(TrainerCallback):
    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs:
            logs["epoch"] = state.epoch
            logs["step"] = state.global_step
            with open("training_log_detailed.csv", "a", encoding="utf-8") as f:
                f.write(json.dumps(logs, ensure_ascii=False) + "\n")

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    compute_metrics=compute_metrics,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=2), LoggingCallback()],
)

if __name__ == "__main__":
    train_result = trainer.train()
    metrics = trainer.state.log_history
    pd.DataFrame(metrics).to_csv("bert_training_log.csv", index=False, encoding="utf-8-sig")
    model.save_pretrained("./bert_reason_classifier")
    tokenizer.save_pretrained("./bert_reason_classifier")