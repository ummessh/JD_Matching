from sklearn.feature_extraction.text import TfidfVectorizer

def build_vectorizer(corpus, max_features=5000):
    """
    Build and fit a TF-IDF vectorizer on the given corpus.
    """
    vectorizer = TfidfVectorizer(max_features=max_features)
    vectorizer.fit(corpus)
    return vectorizer

def vectorize_texts(texts, vectorizer):
    """
    Transform a list of texts into TF-IDF vectors using a fitted vectorizer.
    """
    return vectorizer.transform(texts)

