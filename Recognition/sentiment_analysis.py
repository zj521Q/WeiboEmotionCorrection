import os
import pandas as pd
from keras.models import load_model
import pickle
from keras.utils import pad_sequences
from text_clean import clean_text

main_folder = 'WeiboUserDataset/'
df = pd.DataFrame(columns=['nickname', 'text'])
for user_folder in os.listdir(main_folder):
    user_path = os.path.join(main_folder, user_folder)
    if os.path.isdir(user_path):
        for file_name in os.listdir(user_path):
            file_path = os.path.join(user_path, file_name)
            if os.path.isfile(file_path) and file_name.endswith('.csv'):
                user_df = pd.read_csv(file_path, encoding='utf-8-sig', header=None, names=['text'])
                if not user_df.empty:
                    user_df['nickname'] = user_folder
                    df = pd.concat([df, user_df], ignore_index=True)
print(f"用户总数: {df['nickname'].nunique()}, 总微博条数: {len(df)}")
print(df.sample(10))

df['text1'] = df['text'].fillna('').astype(str)
df['text1'] = df['text1'].apply(clean_text)
df = df[df['text1'].str.split().str.len() > 0]
print(f"清洗后微博条数: {len(df)}")
user_tweet_counts = df['nickname'].value_counts()
valid_users = user_tweet_counts[user_tweet_counts >= 10].index
df = df[df['nickname'].isin(valid_users)]
filtered_user_count = df['nickname'].nunique()
print(f"符合条件的用户数（微博数≥10）: {filtered_user_count}")
print(f"最终微博条数: {len(df)}")
print(df.sample(10))

with open('word_index.pkl', 'rb') as f:
    word_index = pickle.load(f)

model = load_model('model.h5')

max_len = 100
embedding_dim = 100
max_words = len(word_index) + 1
def text_to_sequence(text):
    return [word_index.get(word, 0) for word in text.split()]
sequences = df['text1'].apply(text_to_sequence)
X_new = pad_sequences(sequences, maxlen=max_len, padding='post', truncating='post')

predictions = model.predict(X_new)
predicted_labels = (predictions > 0.5).astype(int).flatten()
df['label'] = predicted_labels
print(df.sample(10))
df.to_csv('weibo_predictions.csv', index=False, encoding='utf-8-sig')
