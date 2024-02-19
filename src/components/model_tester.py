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
    pickle.dump(final_data.to_dict(),open('artifacts/movie_dict.pkl','wb'))
    model_trainer = ModelTrainer()
    
    # Train the model
    similarity_matrix = model_trainer.train_model(final_data)

    # Save the trained model
    with open('artifacts/final_model.pkl', 'wb') as f:
        pickle.dump(similarity_matrix, f)
    logging.info("Trained model saved Successfully")