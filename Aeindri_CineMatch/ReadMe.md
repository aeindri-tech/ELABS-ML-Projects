# CineMatch - Personalized Movie Recommendation Engine

## Project Description
CineMatch is a movie recommendation system built using the MovieLens dataset.  
The system predicts movies that users may like based on their past ratings.

## Algorithms Used
1. Collaborative Filtering using Singular Value Decomposition (SVD)
2. Content-Based Filtering using TF-IDF and Cosine Similarity

## Dataset
MovieLens 100k dataset containing user ratings for movies.

Files used:
- u.data → user ratings
- u.item → movie titles

## Evaluation Metric
RMSE (Root Mean Squared Error) is used to evaluate prediction accuracy.

## Example Features
- Predict movie ratings for users
- Recommend top movies for a user
- Find similar movies based on titles

## Technologies Used
- Python
- Pandas
- NumPy
- Scikit-learn