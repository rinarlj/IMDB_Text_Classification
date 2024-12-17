from models import WordEmbeddingMLP
import pickle


with open('/content/drive/MyDrive/imdb_train_test_text.pkl', 'rb') as f:
    data = pickle.load(f)

    train_text = data['train_text']
    test_text = data['test_text']
    train_labels = data['train_labels']
    test_labels = data['test_labels']


vocab_size = 10000
max_length = 200
embedding_dim = 50

wemlp = WordEmbeddingMLP(vocab_size=vocab_size, max_length=max_length, embedding_dim=embedding_dim)

X_train, y_train = wemlp.preprocess(train_text, train_labels)
X_test, y_test = wemlp.preprocess(test_text, test_labels)

wemlp.build_model()
wemlp.train(X_train, y_train, X_test, y_test, batch_size=128, epochs=20)

accuracy = wemlp.evaluate(X_test, y_test)
print("   ")
print("----------------------------------------------------------------------------------------------")
print("   ")
print(f"Test Accuracy for word embedding + MLP : {accuracy:.2f}")
