import pickle
import streamlit as st
from sklearn.metrics.pairwise import cosine_similarity
from scipy.stats import pearsonr
import numpy as np
import spotipy
import time
from spotipy.oauth2 import SpotifyClientCredentials

# Spotify API credentials
CLIENT_ID = "6829d3f4a1cc41b9862bdb3f2679f3d6"
CLIENT_SECRET = "1e9790b00ceb4ff28529fd9d2bd51060"

# Initialize the Spotify client
client_credentials_manager = SpotifyClientCredentials(client_id=CLIENT_ID, client_secret=CLIENT_SECRET)
sp = spotipy.Spotify(client_credentials_manager=client_credentials_manager)

# Load pre-trained models and sampled DataFrame
w2v_feature_array = pickle.load(open('w2v_feature_array.pkl', 'rb'))
glove_feature_array = pickle.load(open('glove_feature_array.pkl', 'rb'))
doc_embeddings = pickle.load(open('doc_embeddings.pkl', 'rb'))
music = pickle.load(open('sampled_df.pkl', 'rb'))

# Function to recommend similar songs based on Word2Vec similarity
def recommend_word2vec(song, artist):
    start_time = time.time() 
    song_index = music[(music['song'] == song) & (music['artist'] == artist)].index
    if len(song_index) == 0:
        st.error("Song not found. Please try again.")
        return
    else:
        song_index = song_index[0]

    user_song_vector = w2v_feature_array[song_index].reshape(1, -1)
    cosine_scores = cosine_similarity(user_song_vector, w2v_feature_array).flatten()
    jaccard_scores = [get_jaccard_similarity(music['combined_features'][song_index], other_song_text) for other_song_text in music['combined_features']]
    
    # Calculate Pearson scores for each song vector individually
    pearson_scores = [get_pearson_similarity(user_song_vector.flatten(), song_vector.flatten()) for song_vector in w2v_feature_array]
    
    top_cosine = get_top_similar_songs(cosine_scores, music, 11)
    top_jaccard = get_top_similar_songs(jaccard_scores, music, 11)
    top_pearson = get_top_similar_songs(pearson_scores, music, 11)

    end_time = time.time()  # Record end time
    st.write(f"Time taken for Word2Vec recommendation: {end_time - start_time:.2f} seconds")  # Display time taken
    return top_cosine, top_jaccard, top_pearson


# Function to recommend similar songs based on GloVe similarity
def recommend_glove(song, artist):
    start_time = time.time()
    song_index = music[(music['song'] == song) & (music['artist'] == artist)].index
    if len(song_index) == 0:
        st.error("Song not found. Please try again.")
        return
    else:
        song_index = song_index[0]

    user_song_vector = glove_feature_array[song_index].reshape(1, -1)
    cosine_scores = cosine_similarity(user_song_vector, glove_feature_array).flatten()
    jaccard_scores = [get_jaccard_similarity(music['combined_features'][song_index], other_song_text) for other_song_text in music['combined_features']]
    pearson_scores = [pearsonr(user_song_vector.flatten(), song_vector)[0] for song_vector in glove_feature_array]

    top_cosine = get_top_similar_songs(cosine_scores, music, 11)
    top_jaccard = get_top_similar_songs(jaccard_scores, music, 11)
    top_pearson = get_top_similar_songs(pearson_scores, music, 11)
    end_time = time.time()
    st.write(f"Time taken for GloVe recommendation: {end_time - start_time:.2f} seconds")

    return top_cosine, top_jaccard, top_pearson

# Function to recommend similar songs based on Doc2Vec similarity
def recommend_doc2vec(song, artist):
    start_time = time.time()
    song_index = music[(music['song'] == song) & (music['artist'] == artist)].index
    if len(song_index) == 0:
        st.error("Song not found. Please try again.")
        return
    else:
        song_index = song_index[0]

    user_song_vector = doc_embeddings[song_index]
    cosine_scores = [cosine_similarity_vectorized(user_song_vector, song_vector) for song_vector in doc_embeddings]
    jaccard_scores = [get_jaccard_similarity(music['combined_features'][song_index], other_song_text) for other_song_text in music['combined_features']]
    pearson_scores = [pearsonr(user_song_vector, song_vector)[0] for song_vector in doc_embeddings]

    top_cosine = get_top_similar_songs(cosine_scores, music, 11)
    top_jaccard = get_top_similar_songs(jaccard_scores, music, 11)
    top_pearson = get_top_similar_songs(pearson_scores, music, 11)

    end_time = time.time()
    st.write(f"Time taken for Doc2Vec recommendation: {end_time - start_time:.2f} seconds")
    return top_cosine, top_jaccard, top_pearson

# Function to compute Jaccard similarity between two texts
def get_jaccard_similarity(text1, text2):
    set1 = set(text1.split())
    set2 = set(text2.split())
    intersection = len(set1.intersection(set2))
    union = len(set1.union(set2))
    return intersection / union if union != 0 else 0

# Function to compute Pearson correlation coefficient between two vectors
def get_pearson_similarity(vector1, vector2):
    if len(vector1) != len(vector2):
        return 0  # Return 0 if vectors have different lengths
    pearson_corr, _ = pearsonr(vector1.flatten(), vector2.flatten())
    return pearson_corr


# Function to compute cosine similarity between two vectors
def cosine_similarity_vectorized(u, v):
    return np.dot(u, v) / (np.linalg.norm(u) * np.linalg.norm(v))

# Function to get top similar songs
def get_top_similar_songs(scores, music, top_n):
    similar_songs = list(enumerate(scores))
    sorted_similar_songs = sorted(similar_songs, key=lambda x: x[1], reverse=True)[:top_n]
    return [(music.iloc[i]['song'], music.iloc[i]['artist'], score) for i, score in sorted_similar_songs]

# Function to get album cover URL from Spotify API
def get_song_album_cover_url(song_name, artist_name):
    search_query = f"track:{song_name} artist:{artist_name}"
    results = sp.search(q=search_query, type="track")

    if results and results["tracks"]["items"]:
        track = results["tracks"]["items"][0]
        album_cover_url = track["album"]["images"][0]["url"]
        return album_cover_url
    else:
        return "https://i.postimg.cc/0QNxYz4V/social.png"  # Default image if not found

# Function to get Spotify song link
def get_spotify_song_link(song_name, artist_name):
    search_query = f"track:{song_name} artist:{artist_name}"
    results = sp.search(q=search_query, type="track")

    if results and results["tracks"]["items"]:
        track = results["tracks"]["items"][0]
        song_link = track["external_urls"]["spotify"]
        return song_link
    else:
        return "#"  # Default link if not found

# Streamlit UI
st.header('Music Recommender System')

# Streamlit UI
st.write('''
This application utilizes advanced algorithms to recommend similar songs based on the input of a user-selected song and artist. It employs techniques such as Word2Vec, GloVe, and Doc2Vec to analyze song features and compute similarity scores. Users can explore recommendations generated using different algorithms and discover new music based on their preferences.
''')

# Select algorithm
selected_algorithm = st.selectbox("Select an Algorithm:", ('Word2Vec', 'GloVe', 'Doc2Vec'))

# Dropdown for selecting a song
selected_song = st.selectbox("Select a Song:", music['song'].unique())

# Filter artist list based on selected song
artist_list = music[music['song'] == selected_song]['artist'].values
selected_artist = st.selectbox("Select an Artist:", artist_list)

start_time_total = time.time()  # Record start time for total execution

if st.button('Show Recommendation'):
    if selected_algorithm == 'Word2Vec':
        top_cosine, top_jaccard, top_pearson = recommend_word2vec(selected_song, selected_artist)
    elif selected_algorithm == 'GloVe':
        top_cosine, top_jaccard, top_pearson = recommend_glove(selected_song, selected_artist)
    elif selected_algorithm == 'Doc2Vec':
        top_cosine, top_jaccard, top_pearson = recommend_doc2vec(selected_song, selected_artist)

    # Function to display recommendations for a given similarity score
    def display_recommendations(recommendations):
        col_count = 0
        columns = st.columns(5)  # Create 5 columns for displaying images
        for recommended_song, recommended_artist, score in recommendations[1:]:  # Exclude top one
            with columns[col_count % 5]:  # Cycle through columns
                st.text(f'{recommended_song} - {recommended_artist} | Score: {score:.4f}')
                st.image(get_song_album_cover_url(recommended_song, recommended_artist), width=130)
                st.write(f"Spotify Song Link: [Listen on Spotify]({get_spotify_song_link(recommended_song, recommended_artist)})")
                col_count += 1

    if top_cosine:
        st.subheader('Top 10 Recommended Songs using Cosine Similarity:')
        display_recommendations(top_cosine)

    if top_jaccard:
        st.subheader('Top 10 Recommended Songs using Jaccard Similarity:')
        display_recommendations(top_jaccard)

    if top_pearson:
        st.subheader('Top 10 Recommended Songs using Pearson Correlation Coefficient:')
        display_recommendations(top_pearson)

end_time_total = time.time()  # Record end time for total execution
total_time = end_time_total - start_time_total
st.write(f"Total running time: {total_time:.2f} seconds")
