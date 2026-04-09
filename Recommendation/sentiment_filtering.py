import pandas as pd
from keras.models import load_model
import pickle
from keras.utils import pad_sequences
from text_clean import clean_text

df = pd.read_csv('RecommendationCandidateDataset.csv', encoding='utf-8')
df['cleaned_text'] = df['微博正文'].fillna('').apply(clean_text)

with open('word_index.pkl', 'rb') as f:
    word_index = pickle.load(f)
model = load_model('model.h5')

max_len = 100
embedding_dim = 100
max_words = len(word_index) + 1
def text_to_sequence(text):
    return [word_index.get(word, 0) for word in text.split()]
sequences = df['cleaned_text'].apply(text_to_sequence)
X_new = pad_sequences(sequences, maxlen=max_len, padding='post', truncating='post')

predictions = model.predict(X_new)
predicted_labels = (predictions > 0.5).astype(int).flatten()
df['label'] = predicted_labels

df_positive = df[df['label'] == 1]
df_positive.to_csv('RecommendationCandidateDataset2.csv', index=False, encoding='utf-8-sig',columns=['微博正文','cleaned_text'])
