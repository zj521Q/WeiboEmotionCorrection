# Emotion Correction for Weibo Users via Weakly Supervised Learning and Semantic Understanding

The proposed method follows a three-stage framework—"Recognition–Classification–Recommendation" (RCR). First, a deep learning–based model is employed to identify the emotional polarity of Weibo texts. Subsequently, a hybrid method combining lexicon-based weak supervision and BERT modeling is adopted to classify negative posts into corresponding emotion-cause categories. Then, a dual-filtering strategy based on emotional polarity and similarity to users' dominant emotion causes is applied, ultimately recommending positively oriented content aligned with users' specific emotion causes.

## Dependencies

We recommend:
- Python 3.8+
- PyTorch 2.4.1 / Keras 2.10.0

Main packages: `torch`, `transformers`, `pandas`, `numpy`, `scikit-learn`, `jieba`, `sentence-transformers`, `openpyxl`, etc.

## Datasets & Pretrained Models

| Data | Description |
|------|-------------|
| `merged_data.csv` | Train sentiment recognition model |
| `WeiboUserDataset/` | Raw user posts (folder, multiple subdirectories) |
| `Emotion-CauseLexiconCorpusDataset/` | Emotion cause lexicon corpus |
| `fine_tuning.xlsx` | Annotated data for BERT fine-tuning |
| `RecommendationCandidateDataset.csv` | Candidate content for recommendation |

Pretrained models (download from Hugging Face):
- [bert-base-chinese](https://huggingface.co/bert-base-chinese)
- [text2vec-base-chinese](https://huggingface.co/shibing624/text2vec-base-chinese)

## Questions or Feedback?

Open an issue or contact us at: [2250520075@stu.xaut.edu.cn]
