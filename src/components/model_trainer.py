import sys
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from src.exception import CustomException
from src.logger import logging
class ModelTrainer:
    def __init__(self):
        self.vectorizer = TfidfVectorizer()

    def train_model(self, final_data):
        try:
            logging.info("Training model started")

            # Fit the TF-IDF vectorizer on the preprocessed tags
            tags_vectors = self.vectorizer.fit_transform(final_data['tags'])

            logging.info("Computing Cosine-Similarities")
            # Compute cosine similarity matrix
            similarity_matrix = cosine_similarity(tags_vectors, tags_vectors)

            logging.info("Model training completed")
            return similarity_matrix
        
        except Exception as e:
            raise CustomException(sys, e)