import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from scipy.spatial.distance import cdist
import plotly.express as px

# Load your data
data = pd.read_csv("data.csv")

# List of numerical columns to consider for similarity calculations
number_cols = ['valence', 'year', 'acousticness', 'danceability', 'duration_ms', 'energy', 'explicit',
               'instrumentalness', 'key', 'liveness', 'loudness', 'mode', 'popularity', 'speechiness', 'tempo']

# Function to retrieve song data for a given song name
def get_song_data(name, data):
    try:
        return data[data['name'].str.lower() == name].iloc[0]
    except IndexError:
        return None

# Function to calculate the mean vector of a list of songs
def get_mean_vector(song_list, data):
    song_vectors = []
    for song in song_list:
        song_data = get_song_data(song['name'], data)
        if song_data is None:
            print(f"Warning: {song['name']} does not exist in the dataset")
            return None
        song_vector = song_data[number_cols].values
        song_vectors.append(song_vector)
    song_matrix = np.array(list(song_vectors))
    return np.mean(song_matrix, axis=0)

# Function to recommend songs based on a list of seed songs
def recommend_songs(seed_songs, data, n_recommendations=10):
    metadata_cols = ['name', 'artists', 'year']
    song_center = get_mean_vector(seed_songs, data)
    
    # Return an empty list if song_center is missing
    if song_center is None:
        return []
    
    # Normalize the song center
    normalized_song_center = min_max_scaler.transform([song_center])
    
    # Standardize the normalized song center
    scaled_normalized_song_center = standard_scaler.transform(normalized_song_center)
    
    # Calculate Euclidean distances and get recommendations
    distances = cdist(scaled_normalized_song_center, scaled_normalized_data, 'euclidean')
    index = np.argsort(distances)[0]
    
    # Filter out seed songs and duplicates, then get the top n_recommendations
    rec_songs = []
    for i in index:
        song_name = data.iloc[i]['name']
        if song_name not in [song['name'] for song in seed_songs] and song_name not in [song['name'] for song in rec_songs]:
            rec_songs.append(data.iloc[i])
            if len(rec_songs) == n_recommendations:
                break
    
    return pd.DataFrame(rec_songs)[metadata_cols].to_dict(orient='records')

# Normalize the song data using Min-Max Scaler
min_max_scaler = MinMaxScaler()
normalized_data = min_max_scaler.fit_transform(data[number_cols])

# Standardize the normalized data using Standard Scaler
standard_scaler = StandardScaler()
scaled_normalized_data = standard_scaler.fit_transform(normalized_data)

# Streamlit app
st.title('Music Recommender')

st.header('Music Recommender Prompt')

# Input for song names (use st.text_input or st.text_area)
song_names = st.text_area("Enter song names (one per line):")

# Slider to select the number of recommendations
n_recommendations = st.slider("Select the number of recommendations:", 1, 30, 10)

# Convert input to list of song names
input_song_names = song_names.strip().split('\n') if song_names else []

# Button to recommend songs
if st.button('Recommend'):
    # Convert input to list of seed songs
    seed_songs = [{'name': name.lower()} for name in input_song_names]
    
    # Filter out empty names
    seed_songs = [song for song in seed_songs if song['name']]
    
    if not seed_songs:
        st.warning("Please enter at least one song name.")
    else:
        # Call the recommend_songs function
        recommended_songs = recommend_songs(seed_songs, data, n_recommendations)
        
        if not recommended_songs:
            st.warning("No recommendations available based on the provided songs.")
        else:
            # Convert the recommended songs to a DataFrame
            recommended_df = pd.DataFrame(recommended_songs)
            
            # Create a bar plot of recommended songs by name
            recommended_df['text'] = recommended_df.apply(lambda row: f"{row.name + 1}. {row['name']} by {row['artists']} ({row['year']})", axis=1)
            fig = px.bar(recommended_df, y='name', x=range(len(recommended_df), 0, -1), title='Recommended Songs', orientation='h', color='name', text='text')
            fig.update_layout(xaxis_title='Recommendation Rank', yaxis_title='Songs', showlegend=False, uniformtext_minsize=20, uniformtext_mode='show', yaxis_showticklabels=False, height=1000, width=1000)
            fig.update_traces(width=1)
            st.plotly_chart(fig)

st.header('Music Data')

# Display the top songs by popularity
st.subheader('Top Songs by Popularity')
top_songs = data.nlargest(20, 'popularity')
fig_popularity = px.bar(top_songs, x='popularity', y='name', orientation='h',
                        title='Top Songs by Popularity', color='name')
fig_popularity.update_layout(showlegend=False, height=1000, width=1000)
st.plotly_chart(fig_popularity)

# Convert release_date to datetime and extract decade
data['release_date'] = pd.to_datetime(data['release_date'])
data['release_decade'] = (data['release_date'].dt.year // 10) * 10

# Count the number of songs per decade
decade_counts = data['release_decade'].value_counts().sort_index()

# Display the number of songs per decade
st.subheader('Number of Songs per Decade')
fig_decades = px.bar(x=decade_counts.index, y=decade_counts.values,
                     labels={'x': 'Decade', 'y': 'Number of Songs'},
                     title='Number of Songs per Decade', color=decade_counts.values)
fig_decades.update_layout(xaxis_type='category', height=1000, width=1000)
st.plotly_chart(fig_decades)

# Display the distribution of song attributes using a histogram
st.subheader('Distribution of Song Attributes')
attribute_to_plot = st.selectbox('Select an attribute to plot:', number_cols)
fig_histogram = px.histogram(data, x=attribute_to_plot, nbins=30,
                              title=f'Distribution of {attribute_to_plot}')
fig_histogram.update_layout(height=1000, width=1000)
st.plotly_chart(fig_histogram)

# Display a bar plot of artists with the most songs in the dataset
st.subheader('Artists with Most Songs')
top_artists = data['artists'].str.replace("[", "").str.replace("]", "").str.replace("'", "").value_counts().head(20)
fig_top_artists = px.bar(top_artists, x=top_artists.index, y=top_artists.values, color=top_artists.index,
                         labels={'x': 'Artist', 'y': 'Number of Songs'},
                         title='Top Artists with Most Songs')
fig_top_artists.update_xaxes(categoryorder='total descending')
fig_top_artists.update_layout(height=1000, width=1000, showlegend=False)
st.plotly_chart(fig_top_artists)