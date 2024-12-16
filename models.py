from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer



def load_data(train_path, test_path):
    """
    Charge les données d'entraînement et de test à partir des fichiers pickle.
    :param train_path: Chemin du fichier pickle des données d'entraînement.
    :param test_path: Chemin du fichier pickle des données de test.
    :return: Tuple (train_data, train_labels, test_data, test_labels).
    """
    with open(train_path, "rb") as f:
        train_data, train_labels = pickle.load(f)

    with open(test_path, "rb") as f:
        test_data, test_labels = pickle.load(f)

    return train_data, train_labels, test_data, test_labels

class BowLog:
    def __init__(self, max_features=10000):
        self.vectorizer = CountVectorizer(max_features=max_features)
        self.model = LogisticRegression()

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
    def __init__(self, max_features=10000):
        self.vectorizer = TfidfVectorizer(max_features=max_features)
        self.model = LogisticRegression()

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
    def __init__(self, max_features=10000, ngram_range=(1, 2)):
        self.vectorizer = CountVectorizer(max_features=max_features, ngram_range=ngram_range)
        self.model = LogisticRegression()

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