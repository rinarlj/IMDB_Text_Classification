import pickle
from models import NGramLog

import pickle


with open('/content/drive/MyDrive/imdb_train_test_text.pkl', 'rb') as f:
    data = pickle.load(f)
    train_data = data['train_text']
    test_data = data['test_text']
    train_labels = data['train_labels']
    test_labels = data['test_labels']

model = NGramLog(max_features=10000, ngram_sizes=(2, 3, 5))

X_train, X_test = model.preprocess(train_data, test_data)

model.train(X_train, train_labels)

accuracy = model.evaluate(X_test, test_labels)
print("   ")
print("----------------------------------------------------------------------------------------------")
print("   ")
print(f"Accuracy with N-gram: {accuracy:.4f}")

