import os
import pandas as pd


df = pd.read_csv('weibo_predictions.csv', encoding='utf-8-sig')
with open('negative_users.txt', 'r', encoding='utf-8-sig') as f:
    negative_users = [line.strip() for line in f if line.strip()]
neg_df = df[df['nickname'].isin(negative_users)]
pos_weibos = neg_df[neg_df['label'] == 1].copy()
neg_weibos = neg_df[neg_df['label'] == 0].copy()

dict_dir = 'Domainlexicon'
reason_keywords = {}
for fname in os.listdir(dict_dir):
    if not fname.endswith('.csv'):
        continue
    category = os.path.splitext(fname)[0].replace('_lexicon','')
    df_kw = pd.read_csv(os.path.join(dict_dir, fname), encoding='utf-8-sig')
    kws = df_kw['关键词'].dropna().astype(str).str.strip().tolist()
    reason_keywords[category] = kws

def classify_by_count(token_text, reason_dict):
    if isinstance(token_text, str):
        tokens = token_text.split()
    else:
        tokens = token_text
    tokens_lower = [t.lower() for t in tokens]
    counts = {}
    for cat, kws in reason_dict.items():
        kws_lower = [kw.lower() for kw in kws]
        match_num = sum(tokens_lower.count(kw) for kw in kws_lower)
        counts[cat] = match_num
    best_cat, best_count = max(counts.items(), key=lambda x: x[1])
    return best_cat if best_count > 0 else '其他'

neg_weibos['reason'] = neg_weibos['text1'].apply(
    lambda txt: classify_by_count(txt, reason_keywords)
)

user_reason_counts = (
    neg_weibos
    .groupby(['nickname', 'reason'])
    .size()
    .reset_index(name='count')
)

def sorted_reasons(df):
    df_sorted = df.sort_values('count', ascending=False)
    return df_sorted['reason'].tolist()

user_reasons = (
    user_reason_counts
    .groupby('nickname')
    .apply(sorted_reasons)
    .reset_index(name='user_reasons')
)

neg_weibos.to_csv('weibo_negative_with_one_reason.csv', index=False, encoding='utf-8-sig')
user_reasons.to_csv('user_reasons_list_sorted.csv', index=False, encoding='utf-8-sig')