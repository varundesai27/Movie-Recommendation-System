import os
import sys
from dataclasses import dataclass
import pandas as pd
import numpy as np
import ast
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from src.utils import save_object
from src.logger import logging
from src.exception import CustomException

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path: str = os.path.join('artifacts', 'preprocessor.pkl')

class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    # Function to fetch name key's value for genres and keywords
    def convert(self, obj):
        L = []
        for i in ast.literal_eval(obj):
            L.append(i['name'])
        return L
    
    #Function to fetch top 4 actor/actress name
    def convert4(self, obj):
        L = []
        count = 0
        for i in ast.literal_eval(obj):
            if count != 4:
                L.append(i['name'])
                count += 1
            else:
                break
        return L
    
    #Function to only fetch Director name from 'crew'
    def get_director(self, obj):
        L = []
        for i in ast.literal_eval(obj):
            if i['job'] == 'Director':
                L.append(i['name'])
                break
        return L
    
    # Stopword removal, Stemming processing function
    def preprocess(self, text_or_list):
        if isinstance(text_or_list, list):
            # If input is a list, join the elements into a single string
            text = ' '.join(text_or_list)
        else:
            # If input is already a string, use it as is
            text = text_or_list

        tokens = word_tokenize(text)
        tokens = [word for word in tokens if word.isalnum()]

        stop_words = set(stopwords.words('english'))
        tokens = [word for word in tokens if word.lower() not in stop_words]

        stemmer = PorterStemmer()
        tokens = [stemmer.stem(word) for word in tokens]
        processed_text = ' '.join(tokens)

        return processed_text


    def initiate_data_transformation(self, data_path):
        try:
            logging.info("Reading the file in Data Transformation file")
            movies_df = pd.read_csv(data_path)
            
            logging.info("Data Read Successfully")

            movies_df['genres'] = movies_df['genres'].apply(self.convert)
            movies_df['keywords'] = movies_df['keywords'].apply(self.convert)
            movies_df['cast'] = movies_df['cast'].apply(self.convert4)
            movies_df['crew'] = movies_df['crew'].apply(self.get_director)
            movies_df['overview'] = movies_df['overview'].fillna('')
            movies_df['overview'] = movies_df['overview'].apply(lambda x: x.split())

            movies_df['genres'] = movies_df['genres'].apply(lambda x: [i.replace(" ", "") for i in x])
            movies_df['keywords'] = movies_df['keywords'].apply(lambda x: [i.replace(" ", "") for i in x])
            movies_df['cast'] = movies_df['cast'].apply(lambda x: [i.replace(" ", "") for i in x])
            movies_df['crew'] = movies_df['crew'].apply(lambda x: [i.replace(" ", "") for i in x])
            movies_df['tags'] = movies_df['overview'] + movies_df['genres'] + movies_df['keywords'] +  movies_df['cast'] + movies_df['crew']
            
            logging.info("Basic Preprocessing is performed successfully on the data")

            final_df = movies_df[['id', 'title', 'tags']]

            final_df['tags'] = final_df['tags'].apply(lambda x: [item.lower() for item in x])
            final_df['tags'] = final_df['tags'].apply(lambda x: self.preprocess(x))

            logging.info("NLP operations: Stemming, Stopword Removal done")

            save_object(
                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=final_df,
            )

            logging.info("Preprocessed Object Saved successfully")

            return final_df

        except Exception as e:
            raise CustomException(sys, e)


    def load_preprocessed_data(self):
        try:
            return pd.read_pickle(self.data_transformation_config.preprocessor_obj_file_path)
        except FileNotFoundError:
            print("Preprocessed data file not found. Run data transformation first.")
