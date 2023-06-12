import tensorflow as tf
import pickle
import re
import numpy as np

test_model = tf.keras.models.load_model('models/AmaModel10M.h5')

max_sequence_length = 100  # You can define your own max length

# Load vocabulary
with open('vocabulary10M.pkl', 'rb') as f:
    vocab = pickle.load(f)

def clean_text(text):

    # Remove everything except words and spaces
    text = re.sub('[^a-zA-Z\s]', '', text)
    
    # Remove stopwords and convert to lowercase
    text = ' '.join([word.lower() for word in text.split()])
    
    return text

test_reviews = [
    "This product is amazing. It exceeded my expectations and I would definitely recommend it to my friends.",
    "I'm very disappointed with this purchase. The product broke after only a week of use. I wouldn't recommend buying it.",
    "This is by far the best purchase I've made this year! The quality is top-notch and it's great value for money.",
    "I have mixed feelings about this product. On one hand, it's quite useful but on the other hand, it's quite expensive for what it offers.",
    "The product is terrible. It's a waste of money. I regret buying it.",
    "Excellent product. It's exactly what I was looking for. Very happy with my purchase.",
    "The product works as described but the customer service was poor when I had an issue.",
    "Not satisfied at all. This product is clearly overrated. Would not buy again.",
    "This is an average product. Nothing special but not terrible either.",
    "I love this! Works perfectly and is just what I needed. Fast shipping as well."
]


harder_reviews = [
    "The movie 'Inception' is an unforgettable adventure. Directed by Christopher Nolan, the film takes you on a journey through the human mind and dream world. The storyline is complex, requiring the audience's full attention, but it's this complexity that makes the movie all the more rewarding. The film's stunning visuals and impressive special effects only add to the brilliance of the overall storyline. The cast, led by Leonardo DiCaprio, is equally impressive. Each actor brings a certain depth and credibility to their role, making the characters truly memorable. If you're a fan of thought-provoking cinema with a dash of action, 'Inception' is a must-watch.",
    "Pixar's 'Up' is a heartwarming tale of adventure, friendship, and the enduring power of love. The film, which starts on a melancholic note, quickly turns into an exciting adventure featuring floating houses, talking dogs, and a mythical bird. The film is visually stunning, boasting beautiful animation and vibrant colors. The score by Michael Giacchino is touching and fits the emotional tone of the movie perfectly. However, it's the movie's emotional depth that truly sets it apart. 'Up' explores themes of love, loss, and redemption in a way that is both moving and relatable. It's a film that's sure to touch the hearts of both children and adults alike.",
    "Directed by David Fincher, 'Fight Club' is a dark and disturbing film that is not for the faint-hearted. Starring Brad Pitt and Edward Norton, the film explores themes of consumerism, identity, and masculinity. The story, based on the novel of the same name, follows the narrator's descent into madness and the creation of the underground 'fight club'. The film's gritty and stylized visuals complement the disturbing and complex storyline. Pitt and Norton deliver exceptional performances, bringing depth and nuance to their characters. 'Fight Club' is a provocative and challenging film that is sure to leave you questioning your perceptions of reality.",
    "While 'Transformers: Dark of the Moon' delivers in terms of action and special effects, it falls short when it comes to storyline and character development. The film, directed by Michael Bay, is the third installment in the Transformers series. Like its predecessors, it's packed with explosive action sequences and impressive CGI. However, the film's plot feels convoluted and lacks depth. The characters are also underdeveloped, making it hard to connect with them on an emotional level. Despite these flaws, 'Transformers: Dark of the Moon' is a decent popcorn flick if you're just looking for action and visual spectacle."
]


# Transform text into sequence of numbers

# text = "Even tho everyone is saying that they hate this product, I actually love it. It's exactly what I was looking for."

for _ in range(10):
    for text in test_reviews:
        tokens = clean_text(text).split()
        numeric_tokens = [vocab.get(token, vocab['<UNK>']) for token in tokens]
        numeric_padded = numeric_tokens[:max_sequence_length] + [vocab['<PAD>']]*(max_sequence_length-len(numeric_tokens))

        x_test = np.expand_dims(numeric_padded, axis=0)  # Adds an extra dimension
        
        prediction = test_model.predict(x_test, verbose=0)
        best_prediction = np.argmax(prediction) + 1
        prediction_certainity = np.max(prediction)
        print("Text: ", text)
        print("Prediction :", best_prediction, "⭐️ with ", round(prediction_certainity*100, 2), "% confidence")


# tokens = clean_text(text).split()
# numeric_tokens = [vocab.get(token, vocab['<UNK>']) for token in tokens]
# numeric_padded = numeric_tokens[:max_sequence_length] + [vocab['<PAD>']]*(max_sequence_length-len(numeric_tokens))

# x_test = np.expand_dims(numeric_padded, axis=0)  # Adds an extra dimension

# prediction = test_model.predict(x_test)
# print("Text:", text)
# print(prediction)
