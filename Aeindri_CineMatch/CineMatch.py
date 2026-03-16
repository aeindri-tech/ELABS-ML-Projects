import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
ratings=pd.read_csv("ELABS-ML-Projects/Aeindri_CineMatch/u.data",sep="\t",
                    names=["user_id","movie_id","rating","timestamp"])
movies=pd.read_csv("ELABS-ML-Projects/Aeindri_CineMatch/u.item",sep="|",encoding="latin-1",
                   header=None,usecols=[0,1],names=["movie_id","title"])
data=pd.merge(ratings, movies, on="movie_id")
print("\nDataset Loaded Successfully\n")
print(data.head())
movie_matrix=data.pivot_table(index="user_id",columns="title",values="rating")
movie_matrix=movie_matrix.fillna(0)
train,test=train_test_split(ratings, test_size=0.2, random_state=42)
print("\nRunning Collaborative Filtering (SVD)...")
svd = TruncatedSVD(n_components=20, random_state=42)
matrix_reduced = svd.fit_transform(movie_matrix)
matrix_reconstructed = np.dot(matrix_reduced, svd.components_)
predicted_ratings = pd.DataFrame(matrix_reconstructed,index=movie_matrix.index,
                                 columns=movie_matrix.columns)
predictions=[]
actuals=[]
for row in test.itertuples():
    user=row.user_id
    movie=data[data.movie_id == row.movie_id]["title"].values[0]
    if movie in predicted_ratings.columns:
        predictions.append(predicted_ratings.loc[user, movie])
        actuals.append(row.rating)
rmse=np.sqrt(mean_squared_error(actuals, predictions))
print("\nModel Evaluation")
print("RMSE:", rmse)
def recommend_collaborative(user_id, num_recommendations=5):
    user_ratings = predicted_ratings.loc[user_id]
    recommendations = user_ratings.sort_values(ascending=False)
    print("\nTop Recommendations for User", user_id)
    print(recommendations.head(num_recommendations))
print("\n Running Content-Based Filtering...")
tfidf=TfidfVectorizer(stop_words="english")
tfidf_matrix=tfidf.fit_transform(movies["title"])
cosine_sim=cosine_similarity(tfidf_matrix,tfidf_matrix)
indices=pd.Series(movies.index, index=movies["title"]).drop_duplicates()
def recommend_content(title, num_recommendations=5):
    idx=indices[title]
    sim_scores=list(enumerate(cosine_sim[idx]))
    sim_scores=sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores=sim_scores[1:num_recommendations + 1]
    movie_indices=[i[0] for i in sim_scores]
    print("\nMovies similar to:", title)
    print(movies["title"].iloc[movie_indices])
recommend_collaborative(user_id=10)
recommend_content("Toy Story (1995)")