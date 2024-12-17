from models import LSTMClassifier
import pickle

with open('/content/drive/MyDrive/imdb_train_test_text.pkl', 'rb') as f:
    data = pickle.load(f)

train_text = data['train_text']
test_text = data['test_text']
train_labels = data['train_labels']
test_labels = data['test_labels']

vocab_size = 10000
max_length = 200
embedding_dim = 100
lstm_units = 256

lstm_classifier = LSTMClassifier(vocab_size=vocab_size, max_length=max_length, embedding_dim=embedding_dim, lstm_units=lstm_units)

X_train, y_train = lstm_classifier.preprocess(train_text, train_labels)
X_test, y_test = lstm_classifier.preprocess(test_text, test_labels)


lstm_classifier.build_model() 
lstm_classifier.train(X_train, y_train, X_test, y_test, batch_size=128, epochs=20)

accuracy = lstm_classifier.evaluate(X_test, y_test)
print("   ")
print("----------------------------------------------------------------------------------------------")
print("   ")
print(f"Test Accuracy: {accuracy:.2f}")
