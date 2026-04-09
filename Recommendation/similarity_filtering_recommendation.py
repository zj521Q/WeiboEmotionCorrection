import os
import pandas as pd
import ast
import torch
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from tqdm import tqdm
from sentence_transformers import SentenceTransformer, util

BATCH_SIZE = 128
ALPHA = 0.7
BETA = 0.3
TOTAL_RECS_PER_USER = 10
TOP_K_CACHE_PER_REASON = 100
def min_max_scale(scores: np.ndarray) -> np.ndarray:
    min_score, max_score = np.min(scores), np.max(scores)
    if max_score == min_score:
        return np.zeros_like(scores)
    return (scores - min_score) / (max_score - min_score)

try:
    user_df_full = pd.read_csv('user_reasons_list_sorted2_BERT.csv')
    dict_folder = 'Domainlexicon'
    reason_dicts = {
        fname.replace('_lexicon.csv', ''): pd.read_csv(os.path.join(dict_folder, fname))['关键词']
        .dropna()
        .astype(str)
        .tolist()
        for fname in os.listdir(dict_folder)
        if fname.endswith('_lexicon.csv')
    }
    all_keywords = [kw for kws in reason_dicts.values() for kw in kws]
    weibo_df = pd.read_csv('RecommendationCandidateDataset2.csv', encoding="utf-8")
    user_weibo_df_full = pd.read_csv('weibo_negative_with_one_reason_BERT.csv')
except FileNotFoundError as e:
    print(f"找不到必需的数据文件 -> {e}")
    exit()

user_df_full['reasons_list'] = user_df_full['user_reasons'].apply(ast.literal_eval)
user_df_full['reasons_no_other'] = user_df_full['reasons_list'].apply(lambda lst: [r for r in lst if r != '其他'])

weibo_df['cleaned_text'] = weibo_df['cleaned_text'].astype(str)
weibo_df['weibo_text'] = weibo_df['微博正文'].astype(str)
cleaned_texts = weibo_df['cleaned_text'].tolist()

user_weibo_df_full['text1'] = user_weibo_df_full['text1'].astype(str)
user_df = user_df_full
user_weibo_df = user_weibo_df_full

vectorizer = TfidfVectorizer(token_pattern=r'(?u)\b\w+\b')
vectorizer.fit(all_keywords + cleaned_texts)
keyword_vecs = vectorizer.transform(all_keywords)
weibo_vecs_tfidf = vectorizer.transform(cleaned_texts)
tfidf_sim_matrix = cosine_similarity(keyword_vecs, weibo_vecs_tfidf)
reason_labels = [reason for reason, kws in reason_dicts.items() for kw in kws]
reason_to_idxs = {reason: [i for i, r_label in enumerate(reason_labels) if r_label == reason] for reason in reason_dicts}
sims_cause_tfidf_dict = {reason: tfidf_sim_matrix[idxs, :].max(axis=0) for reason, idxs in reason_to_idxs.items()}

model = SentenceTransformer('text2vec-base-chinese')
device = "cuda" if torch.cuda.is_available() else "cpu"
model = model.to(device)
print(f"模型已加载到 {device} (batch_size={BATCH_SIZE})")

weibo_embeddings = model.encode(
    cleaned_texts,
    convert_to_tensor=True,
    show_progress_bar=True,
    device=device,
    batch_size=BATCH_SIZE
)

user_personal_embeds = {}
for nickname, group in tqdm(user_weibo_df.groupby('nickname'), desc="编码用户个人画像"):
    texts = group['text1'].tolist()
    if not texts:
        continue
    emb = model.encode(
        texts,
        convert_to_tensor=True,
        show_progress_bar=False,
        device=device,
        batch_size=BATCH_SIZE
    )
    user_personal_embeds[nickname] = emb.mean(dim=0, keepdim=True)

sims_personal_sbert_dict = {
    nick: util.cos_sim(p_emb, weibo_embeddings).cpu().numpy().flatten()
    for nick, p_emb in user_personal_embeds.items()
}

user_reason_topk_cache = {}
for _, row in tqdm(user_df.iterrows(), total=len(user_df), desc="预计算每个用户-原因TopK"):
    nickname = row['nickname']
    if nickname not in user_personal_embeds:
        continue

    sims_personal_sbert = sims_personal_sbert_dict[nickname]
    norm_personal_sbert = min_max_scale(sims_personal_sbert)
    user_reason_topk_cache[nickname] = {}

    for reason in row['reasons_no_other']:
        if reason not in sims_cause_tfidf_dict:
            continue
        sims_cause_tfidf = sims_cause_tfidf_dict[reason]
        norm_cause_tfidf = min_max_scale(sims_cause_tfidf)
        sims_final = ALPHA * norm_cause_tfidf + BETA * norm_personal_sbert
        sorted_indices = np.argsort(sims_final)[::-1][:TOP_K_CACHE_PER_REASON]
        top_recs_df = weibo_df.iloc[sorted_indices].copy()
        top_recs_df['score'] = sims_final[sorted_indices]
        user_reason_topk_cache[nickname][reason] = top_recs_df


records = []
for _, row in tqdm(user_df.iterrows(), total=len(user_df), desc="生成最终推荐列表"):
    nickname = row['nickname']
    reasons = row['reasons_no_other']
    if nickname not in user_reason_topk_cache:
        continue

    try:
        counts = ast.literal_eval(row['reason_counts'])
        if len(counts) != len(row['reasons_list']):
            counts = [1] * len(row['reasons_list'])
    except Exception:
        counts = [1] * len(row['reasons_list'])

    reason_counts_map = {r: c for r, c in zip(row['reasons_list'], counts) if r != '其他'}
    total_count = sum(reason_counts_map.values())
    if total_count == 0:
        continue

    weights = [reason_counts_map.get(r, 0) / total_count for r in reasons]
    reason_num = [max(0, round(TOTAL_RECS_PER_USER * w)) for w in weights]
    diff = TOTAL_RECS_PER_USER - sum(reason_num)
    if diff != 0:
        sorted_indices = np.argsort(weights)[::-1] if diff > 0 else np.argsort(weights)
        for i in sorted_indices[:abs(diff)]:
            reason_num[i] += np.sign(diff)

    for reason, n_rec in zip(reasons, reason_num):
        if n_rec <= 0 or reason not in user_reason_topk_cache[nickname]:
            continue
        candidate_recs = user_reason_topk_cache[nickname][reason]
        selected_recs = candidate_recs.head(n_rec)
        for _, rec in selected_recs.iterrows():
            records.append({
                'target_user': nickname,
                'matched_reason': reason,
                'weibo_text': rec['weibo_text'],
                'score': rec['score']
            })

out_df = pd.DataFrame(records)
out_df.to_csv('User_Level_Recommendations', index=False, encoding='utf-8-sig')
