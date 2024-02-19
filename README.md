# Movie Recommendation System

This is a movie recommendation system built using content-based filtering. It suggests similar movies to a given input movie using natural language processing techniques.

## Installation

1. Clone the repository:
2. Navigate to the project directory:
3. Install the required dependencies:
    ```bash
    pip install -r requirements.txt
    ```

## Usage

1. Run the model tester script to train the recommendation model and generate necessary files:
    ```bash
    python src/components/model_tester.py
    ```
2. Once the model has been trained and necessary files have been generated, you can run the Streamlit app:

    ```bash
    streamlit run app.py
    ```

## Project Structure

The project is structured as follows:

- `data_ingestion.py`: Module for data ingestion, merging datasets, and storing the merged dataset.
- `data_transformation.py`: Module for data transformation, including preprocessing text data.
- `model_trainer.py`: Module for training the recommendation model.
- `model_tester.py`: Script to execute all project modules and train the model.
- `app.py`: Streamlit front-end application for the movie recommendation system.
