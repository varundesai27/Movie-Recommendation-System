from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer
from pickle4 import pickle
from src.logger import logging

if __name__ == "__main__":
    # Initialize DataIngestion class
    data_ingestion = DataIngestion()
    
    # Fetch merged data
    merged_data = data_ingestion.initiate_data_ingestion()

    # Initialize DataTransformation class
    data_transformation = DataTransformation()
    
    # Transform the merged data
    final_data = data_transformation.initiate_data_transformation(merged_data)
    #print(final_data)
    model_trainer = ModelTrainer()
    
    # Train the model
    similarity_matrix = model_trainer.train_model(final_data)

    # Save the trained model
    with open('artifacts/final_model.pkl', 'wb') as f:
        pickle.dump(similarity_matrix, f)
    logging.info("Trained model saved Successfully")

    # Find the index of the movie "Batman" in the preprocessed data
    # batman_index = final_data[final_data['title'] == 'Batman'].index[0]
    # batman_similarity_scores = similarity_matrix[batman_index]
    # similar_movie_indices = batman_similarity_scores.argsort()[::-1][1:6]
    # similar_movies = final_data.iloc[similar_movie_indices]['title'].values
    # print("Movies similar to 'Batman':")
    # for movie in similar_movies:
    #     print(movie)