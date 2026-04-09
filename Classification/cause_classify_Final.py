import json
import pandas as pd
import torch
import torch.nn.functional as F
from transformers import BertTokenizer, BertForSequenceClassification

df = pd.read_csv('weibo_predictions.csv', encoding='utf-8-sig')
with open('negative_users.txt', 'r', encoding='utf-8-sig') as f:
    negative_users = [line.strip() for line in f if line.strip()]
neg_df = df[df['nickname'].isin(negative_users)]
pos_weibos = neg_df[neg_df['label'] == 1].copy()
neg_weibos = neg_df[neg_df['label'] == 0].copy()

model_path = "./bert_reason_classifier_finetuned"
tokenizer = BertTokenizer.from_pretrained(model_path)
model = BertForSequenceClassification.from_pretrained(model_path)
model.eval()

device = torch.device("cpu")
model.to(device)

with open("label_map.json", "r", encoding="utf-8") as f:
    label_map = json.load(f)
id2label = {v: k for k, v in label_map.items()}
valid_labels = set(label_map.keys())

def batch_classify(texts, tokenizer, model, id2label, batch_size=64, threshold=0.5):
    preds = []
    total = len(texts)
    for i in range(0, total, batch_size):
        batch_texts = texts[i:i + batch_size]
        inputs = tokenizer(
            batch_texts,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=128
        ).to(device)
        with torch.no_grad():
            outputs = model(**inputs)
            probs = F.softmax(outputs.logits, dim=-1)
            confs, batch_preds = torch.max(probs, dim=-1)
        for conf, pred in zip(confs, batch_preds):
            conf = conf.item()
            pred = int(pred.item())
            label = id2label.get(pred, "其他")
            if conf < threshold or label not in valid_labels:
                label = "其他"
            preds.append(label)
        print(f"已处理 {min(i + batch_size, total)} / {total} 条微博...")
    return preds

texts = neg_weibos["text1"].astype(str).tolist()
neg_weibos["reason"] = batch_classify(texts, tokenizer, model, id2label, batch_size=128, threshold=0.6)

user_reason_counts = (
    neg_weibos
    .groupby(["nickname", "reason"])
    .size()
    .reset_index(name="count")
)

def sorted_reasons_and_counts(df):
    df_sorted = df.sort_values("count", ascending=False)
    reasons = df_sorted["reason"].tolist()
    counts = df_sorted["count"].tolist()
    return pd.Series({
        "user_reasons": str(reasons),
        "reason_counts": str(counts)
    })

user_reasons = (
    user_reason_counts
    .groupby("nickname")
    .apply(sorted_reasons_and_counts)
    .reset_index()
)

neg_weibos.to_csv("weibo_negative_with_one_reason_BERT.csv", index=False, encoding='utf-8-sig')
user_reasons.to_csv("user_reasons_list_sorted_BERT.csv", index=False, encoding='utf-8-sig')