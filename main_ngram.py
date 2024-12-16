import pickle
from models import NGramLog

import pickle


with open('/content/drive/MyDrive/imdb_train_test_text.pkl', 'rb') as f:
    data = pickle.load(f)
    train_data = data['train_text']
    test_data = data['test_text']
    train_labels = data['train_labels']
    test_labels = data['test_labels']

# Initialiser le modèle BoW
model = NGramLog(max_features=10000, ngram_sizes=(2, 3, 5))

# Prétraitement
X_train, X_test = model.preprocess(train_data, test_data)

# Entraîner le modèle
model.train(X_train, train_labels)

# Évaluer le modèle
accuracy = model.evaluate(X_test, test_labels)
print(f"Accuracy with Bag of Words: {accuracy:.4f}")

