import pickle
import numpy as np
import pandas as pd
from gensim.models import Word2Vec
from keras import Sequential
from keras.layers import (Embedding, SpatialDropout1D, LSTM,
                          Dense, GlobalMaxPooling1D, Bidirectional)
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from keras.optimizers import Adam
from keras.utils import pad_sequences
from sklearn.model_selection import train_test_split
from text_clean import clean_text

df = pd.read_csv('merged_data.csv', encoding='utf-8')
df = df[['text', 'label']]
df['text'] = df['text'].apply(clean_text)
df = df[df['text'].str.split().str.len() > 0].reset_index(drop=True)

max_len = 100
embedding_dim = 100
with open('word_index.pkl', 'rb') as f:
    word_index = pickle.load(f)
w2v_model = Word2Vec.load('word2vec.model')
max_words = len(word_index) + 1
embedding_matrix = np.zeros((max_words, embedding_dim))
for word, idx in word_index.items():
    if word in w2v_model.wv:
        embedding_matrix[idx] = w2v_model.wv[word]

def text_to_sequence(text):
    return [word_index.get(word, 0) for word in text.split()]
df['seq'] = df['text'].apply(text_to_sequence)
X = pad_sequences(df['seq'], maxlen=max_len, padding='post', truncating='post')
y = df['label'].values

X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.30, random_state=42, stratify=y)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.50, random_state=42, stratify=y_temp)

model = Sequential()
model.add(Embedding(input_dim=max_words,
                    output_dim=embedding_dim,
                    weights=[embedding_matrix],
                    input_length=max_len,
                    trainable=True))
model.add(SpatialDropout1D(0.2))
model.add(Bidirectional(LSTM(128, return_sequences=True, dropout=0.3, recurrent_dropout=0.3)))
model.add(GlobalMaxPooling1D())
model.add(Dense(1, activation='sigmoid'))
optimizer = Adam(learning_rate=1e-3)
model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])
model.summary()
callbacks = [
    EarlyStopping(monitor='val_loss', patience=5, min_delta=1e-4, restore_best_weights=True),
    ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2, min_lr=1e-5)
]
model.fit(
    X_train, y_train,
    epochs=10,
    batch_size=128,
    validation_data=(X_val, y_val),
    callbacks=callbacks
)
model.save('model.h5')
