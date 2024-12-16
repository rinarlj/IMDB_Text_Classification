import pickle
from models import TFIDFLog
from keras.datasets import imdb

# Charger les données IMDB
(train_data, train_labels), (test_data, test_labels) = imdb.load_data(num_words=10000)

# Décodage des indices en texte brut
word_index = imdb.get_word_index()
reverse_word_index = {value: key for key, value in word_index.items()}

def decode_review(encoded_review):
    return " ".join([reverse_word_index.get(i - 3, "?") for i in encoded_review])

train_data = [decode_review(review) for review in train_data]
test_data = [decode_review(review) for review in test_data]

# Initialiser le modèle TF-IDF
model = TFIDFLog(max_features=10000)


# Prétraitement
X_train, X_test = model.preprocess(train_data, test_data)

# Entraîner le modèle
model.train(X_train, train_labels)

# Évaluer le modèle
accuracy = model.evaluate(X_test, test_labels)
print(f"Accuracy with Bag of Words: {accuracy:.4f}")

# Sauvegarder le modèle et le vectoriseur
with open("bow_model.pkl", "wb") as f:
    pickle.dump(model, f)
