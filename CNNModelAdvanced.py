from keras.models import Sequential
from keras.layers import Embedding, Conv1D, GlobalMaxPooling1D, Dense, Dropout, LSTM, Bidirectional

def build_model(vocabulary):
    vocab_size = len(vocabulary) + 1  # Adding 1 because of reserved 0 index
    embed_dim = 128
    max_length = 100

    model = Sequential()
    model.add(Embedding(vocab_size, embed_dim, input_length=max_length))
    model.add(Bidirectional(LSTM(64, return_sequences=True)))
    model.add(GlobalMaxPooling1D())
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(32, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(5, activation='softmax'))  # 5 classes

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model
