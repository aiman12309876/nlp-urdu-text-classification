from sklearn.feature_extraction.text import TfidfVectorizer
from gensim.models import Word2Vec
import numpy as np

def create_tfidf_features(X_train, X_test, max_features=10000):
    vectorizer = TfidfVectorizer(max_features=max_features)
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)
    return X_train_vec, X_test_vec, vectorizer

def create_word2vec_embeddings(sentences, vector_size=100, window=5, min_count=2):
    model = Word2Vec(sentences, vector_size=vector_size, window=window, min_count=min_count)
    return model