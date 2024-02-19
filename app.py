import streamlit as st
import requests
from pickle4 import pickle
import pandas as pd

TMDB_API_KEY = '92fce65c3ebd11a486dfad05e719e1e1'

def fetch_movie_poster(movie_id):
    url = f'https://api.themoviedb.org/3/movie/{movie_id}?api_key={TMDB_API_KEY}&language=en-US'
    response = requests.get(url)
    if response.status_code == 200:
        data = response.json()
        poster_path = data.get('poster_path')
        if poster_path:
            poster_url = f'https://image.tmdb.org/t/p/w500/{poster_path}'
            return poster_url
    return None

def recommend(movie):
    movie_index = movies[movies['title'] == movie].index[0]
    similarity_scores = similarity_matrix[movie_index]
    similar_movie_indices = similarity_scores.argsort()[::-1][1:6]
    similar_movies = movies.iloc[similar_movie_indices]
    
    return similar_movies

movies_list = pickle.load(open('artifacts/movie_dict.pkl', 'rb'))
movies = pd.DataFrame(movies_list)
st.title("Movie Recommendation System")

similarity_matrix = pickle.load(open('artifacts/final_model.pkl', 'rb'))

selected_movie_name = st.selectbox(
    '',
    ['Select or Type Movie here'] + list(movies['title'].values))

if st.button('Recommend'):
    st.write(selected_movie_name)
    similar_movies = recommend(selected_movie_name)
    columns = st.columns(5)
    for col, (movie_id, title) in zip(columns, zip(similar_movies['id'], similar_movies['title'])):
        poster_url = fetch_movie_poster(movie_id)
        if poster_url:
            col.image(poster_url, caption=title, use_column_width=True)
        else:
            col.write(f"No poster found for {title}")
