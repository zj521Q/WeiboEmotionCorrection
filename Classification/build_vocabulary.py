import os
import pandas as pd
import numpy as np
import jieba.posseg as pseg
from sklearn.feature_extraction.text import TfidfVectorizer

top_n = 500
preproc_dir = "Preprocessed_Results"
dict_dir = "Domainlexicon"
os.makedirs(dict_dir, exist_ok=True)

def valid_pos(word):
    words = list(pseg.cut(word))
    if not words:
        return False
    flag = words[0].flag
    if flag.startswith("n"):
        return flag in ["n", "nt", "nz"]
    return flag.startswith(("v", "a"))

for filename in os.listdir(preproc_dir):
    if not filename.endswith(".csv"):
        continue

    domain_name = os.path.splitext(filename)[0]
    df = pd.read_csv(os.path.join(preproc_dir, filename), encoding="utf-8-sig", dtype=str)

    texts = df["处理后文本"].dropna().str.strip().tolist()
    texts = [t for t in texts if t]

    if not texts:
        print(f"[跳过] {domain_name}：无有效文本。")
        continue

    vectorizer = TfidfVectorizer(
        token_pattern=r"(?u)\b\w{2,}\b",
        max_df=0.8,
        min_df=2,
        ngram_range=(1, 1),
        smooth_idf=True,
        sublinear_tf=True
    )
    tfidf = vectorizer.fit_transform(texts)
    vocab = vectorizer.get_feature_names_out()
    scores = np.array(tfidf.sum(axis=0)).flatten()

    pairs = sorted(zip(vocab, scores), key=lambda x: x[1], reverse=True)
    top_pairs = pairs[:top_n]

    valid_words = [w for w, wt in top_pairs if valid_pos(w)]

    if not valid_words:
        print(f"[警告] {domain_name} 过滤后无符合要求的词语，跳过保存词典。")
        continue

    out_df = pd.DataFrame({"关键词": valid_words})
    out_file = os.path.join(dict_dir, f"{domain_name}_lexicon.csv")
    out_df.to_csv(out_file, index=False, encoding="utf-8-sig")
    print(f"[完成] {domain_name} 词典：{len(out_df)} 个关键词，已保存到 {out_file}")