import pandas as pd
import gzip
import tensorflow as tf
from collections import Counter
import nltk
import pickle


count = 0
word_counter = Counter()

chunksize = 10 ** 6
with gzip.open('data/All_Amazon_Review.json.gz', 'r') as file_in:
    for chunk in pd.read_json(file_in, lines=True, chunksize=chunksize): # type: ignore
        count += 1
        chunk = chunk.drop(['verified', 'reviewTime', 'reviewerID', 'reviewerName', 'asin', 'unixReviewTime', 'image', 'style', 'vote', 'summary'], axis=1)
        chunk = chunk[chunk['reviewText'].apply(lambda x: isinstance(x, str))]
        chunk['reviewText'] = chunk['reviewText'].str.replace('[^a-zA-Z\s]', '', regex=True).str.lower()
        chunk['tokens'] = chunk['reviewText'].str.split()
        for _, row in chunk.iterrows():
            word_counter.update(row['tokens'])
        print("Processed", chunksize * count / 10 ** 6, "M lines.")
        if count == 100:
            break
        
 

special_tokens = ['<UNK>', '<PAD>', '<START>', '<END>']

# Create vocabulary
vocabulary = {word: i for i, (word, count) in enumerate(word_counter.most_common(10000))}

# Add special tokens
for token in special_tokens:
    vocabulary[token] = len(vocabulary)


# Save vocabulary
with open('vocabulary10M.pkl', 'wb') as f:
    pickle.dump(vocabulary, f)