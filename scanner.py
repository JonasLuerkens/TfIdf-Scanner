from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

class Scanner:
    def __init__(self):
        self.vectorizer = TfidfVectorizer(stop_words=None)
        self.tfidf_matrix = None
        self.documents = []
        self.doc_ids = []

    def train(self, documents, doc_ids):
        # Train tfidf model with the provided documents
        self.documents = documents
        self.doc_ids = doc_ids
        
        # Calculating tfidf matrix
        self.tfidf_matrix = self.vectorizer.fit_transform(documents)
        
    def find_similars(self, query, threshold):
        # find_similars finds documents similar to the query above the given percentile threshold
        
        # Transform query into a tfidf vector
        query_vector = self.vectorizer.transform([query])

        # Calculate cosine similarities
        similarities = cosine_similarity(query_vector, self.tfidf_matrix).flatten()
        
        # Calculate threshold 
        threshold_value = np.percentile(similarities, threshold)
        
        candidates = []
        
        # Filter documents
        for idx, score in enumerate(similarities):
            if score >= threshold_value and score > 0: # score > 0 filter out docs that are not similar at all
                candidates.append({
                    'id': self.doc_ids[idx],
                    'score': score,
                    'preview': self.documents[idx][:200]
                })

        # Sort candidates by score descending
        candidates.sort(key=lambda x: x['score'], reverse=True)
        
        return candidates, threshold_value