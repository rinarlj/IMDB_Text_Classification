from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.feature_extraction.text import TfidfVectorizer


class BowLog:
    def __init__(self, max_features=10000, max_iter=1000):
        self.vectorizer = CountVectorizer(max_features=max_features)
        self.model = LogisticRegression(max_iter=max_iter)

    def preprocess(self, train_data, test_data):
        """Transforme les données en utilisant BoW."""
        X_train = self.vectorizer.fit_transform(train_data)
        X_test = self.vectorizer.transform(test_data)
        return X_train, X_test

    def train(self, X_train, y_train):
        """Entraîne le modèle."""
        self.model.fit(X_train, y_train)

    def evaluate(self, X_test, y_test):
        """Évalue le modèle sur des données de test."""
        y_pred = self.model.predict(X_test)
        return accuracy_score(y_test, y_pred)

class TFIDFLog:
    def __init__(self, max_features=10000, max_iter=1000):
        self.vectorizer = TfidfVectorizer(max_features=max_features)
        self.model = LogisticRegression(max_iter=max_iter)

    def preprocess(self, train_data, test_data):
        """Transforme les données en utilisant TF-IDF."""
        X_train = self.vectorizer.fit_transform(train_data)
        X_test = self.vectorizer.transform(test_data)
        return X_train, X_test

    def train(self, X_train, y_train):
        """Entraîne le modèle."""
        self.model.fit(X_train, y_train)

    def evaluate(self, X_test, y_test):
        """Évalue le modèle sur des données de test."""
        y_pred = self.model.predict(X_test)
        return accuracy_score(y_test, y_pred)

class NGramLog:
    def __init__(self, max_features=10000, ngram_range=(1, 2), max_iter=1000):
        self.vectorizer = CountVectorizer(max_features=max_features, ngram_range=ngram_range)
        self.model = LogisticRegression(max_iter=max_iter)

    def preprocess(self, train_data, test_data):
        """Transforme les données en utilisant N-Gram."""
        X_train = self.vectorizer.fit_transform(train_data)
        X_test = self.vectorizer.transform(test_data)
        return X_train, X_test

    def train(self, X_train, y_train):
        """Entraîne le modèle."""
        self.model.fit(X_train, y_train)

    def evaluate(self, X_test, y_test):
        """Évalue le modèle sur des données de test."""
        y_pred = self.model.predict(X_test)
        return accuracy_score(y_test, y_pred)