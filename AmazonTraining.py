import pandas as pd
import gzip
import tensorflow as tf
from CNNModelAdvanced import build_model
import pickle
from sklearn.model_selection import train_test_split
import numpy as np
from keras.utils import to_categorical


# Load vocabulary
with open('vocabulary10M.pkl', 'rb') as f:
    vocab = pickle.load(f)


count = 0
model = build_model(vocab)
max_sequence_length = 100  # You can define your own max length

chunksize = 10 ** 6
with gzip.open('data/All_Amazon_Review.json.gz', 'r') as file_in:
    for chunk in pd.read_json(file_in, lines=True, chunksize=chunksize): # type: ignore
        count += 1
        chunk = chunk.drop(['verified', 'reviewTime', 'reviewerID', 'reviewerName', 'asin', 'unixReviewTime', 'image', 'style', 'vote', 'summary'], axis=1)
        chunk = chunk[chunk['reviewText'].apply(lambda x: isinstance(x, str))]
        chunk['reviewText'] = chunk['reviewText'].str.replace('[^a-zA-Z\s]', '', regex=True).str.lower()
        chunk['tokens'] = chunk['reviewText'].str.split()
        chunk['numeric_tokens'] = chunk['tokens'].apply(lambda tokens: [vocab.get(token, vocab['<UNK>']) for token in tokens])
        chunk['numeric_padded'] = chunk['numeric_tokens'].apply(lambda tokens: tokens[:max_sequence_length] + [vocab['<PAD>']]*(max_sequence_length-len(tokens)))
        X = chunk['numeric_padded'].to_list()
        y = chunk['overall'].to_list()
        X = np.array(X)
        y = np.array(y)

        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2)

        y_train_categorical = to_categorical(y_train - 1)  # assuming y_train is 1-indexed (1-5), subtract 1 to make it 0-indexed (0-4)
        y_val_categorical = to_categorical(y_val - 1)

        train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train_categorical))

        val_dataset = tf.data.Dataset.from_tensor_slices((X_val, y_val_categorical))

        # Further batch the data
        train_dataset = train_dataset.batch(32)
        val_dataset = val_dataset.batch(32)
        
        model.fit(train_dataset, epochs=1, validation_data=val_dataset)
        print("Processed", chunksize * count / 10 ** 6, "M reviews.")

        if count == 10:
            break

model.save('models/AmaModel10M.h5')
