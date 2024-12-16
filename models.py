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
    def __init__(self, max_features=10000, ngram_sizes=(2, 3, 5), max_iter=1000):
        self.ngram_sizes = ngram_sizes
        self.vectorizer = CountVectorizer(max_features=max_features, ngram_range=(min(ngram_sizes), max(ngram_sizes)))
        self.model = LogisticRegression(max_iter=max_iter)
    
    def filter_ngrams(self, X, feature_names):
        """Filtre les N-grams pour inclure uniquement ceux ayant des longueurs spécifiques."""
        filtered_features = [
            i for i, feature in enumerate(feature_names)
            if len(feature.split()) in self.ngram_sizes
        ]
        return X[:, filtered_features]

    def preprocess(self, train_data, test_data):
        """Transforme les données et filtre les N-grams."""
        # Crée la matrice avec tous les N-grams dans la plage
        X_train_full = self.vectorizer.fit_transform(train_data)
        X_test_full = self.vectorizer.transform(test_data)
        
        # Filtre les N-grams non désirés
        feature_names = self.vectorizer.get_feature_names_out()
        X_train = self.filter_ngrams(X_train_full, feature_names)
        X_test = self.filter_ngrams(X_test_full, feature_names)
        
        return X_train, X_test

    def train(self, X_train, y_train):
        """Entraîne le modèle."""
        self.model.fit(X_train, y_train)

    def evaluate(self, X_test, y_test):
        """Évalue le modèle sur des données de test."""
        y_pred = self.model.predict(X_test)
        return accuracy_score(y_test, y_pred)