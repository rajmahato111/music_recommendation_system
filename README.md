# Music Recommender System

The Music Recommender System is a project aimed at providing personalized song recommendations to users based on their input seed songs. It utilizes a combination of data analysis, machine learning, and visualization techniques to offer users a curated list of songs that are similar to their chosen seed songs.

## Overview

The recommender system works by calculating the similarity between songs using various numerical attributes such as valence, acousticness, danceability, and more. Users can input the names of their favorite songs, and the system will generate a list of recommended songs that are likely to be enjoyed based on the similarities in these attributes.

## Approach

The project follows a series of steps to achieve the goal of providing accurate and relevant song recommendations:

1. **Data Preprocessing**: The raw song data, including attributes like popularity, energy, and artists, is loaded and processed. Attributes are standardized and normalized to ensure consistency and meaningful comparisons.

2. **Calculating Song Similarities**: The recommender system calculates the similarity between songs using the Euclidean distance metric. The closer two songs are in terms of their attribute values, the more similar they are considered to be.

3. **Creating Seed Songs**: Users input their favorite songs as seed songs. These seed songs are used to determine the user's musical preferences.

4. **Generating Recommendations**: The system generates a list of recommended songs by identifying songs that are similar to the seed songs. It filters out duplicate and seed songs to ensure variety in recommendations.

5. **Visualization**: The recommended songs are displayed in a user-friendly format using a bar plot. Users can see the names of the songs, artists, and release years.

6. **Deployment Using Streamlit**: The project is deployed as a user-friendly web app using Streamlit. Users can input their seed songs, receive recommendations, and visualize the results all through a simple web interface.

## Technologies Used

- Python: For data analysis, processing, and modeling.
- Pandas: For data manipulation and preprocessing.
- Scikit-learn: For data scaling and similarity calculation.
- Plotly: For interactive and visually appealing visualizations.
- Streamlit: For building the user-friendly web app.

## Links

- [Kaggle Notebook](https://www.kaggle.com/aliessamali/music-recommendation-system-streamlit)
- [Streamlit Web App](https://musicrecommendationsystem-zkaftqsquighe8wizjvcas.streamlit.app)

Feel free to contribute, provide feedback, or report issues by opening a GitHub issue or pull request.
