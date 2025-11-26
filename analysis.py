import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler
import joblib


# ----------------------------------------
# Load the main rating data
# ----------------------------------------
ratings_cols = ['user_id', 'movie_id', 'rating', 'timestamp']
ratings = pd.read_csv("u.data", sep="\t", names=ratings_cols)

# ----------------------------------------
# Load movie information
# ----------------------------------------
items_cols = [
    'movie_id', 'movie_title', 'release_date', 'video_release_date',
    'IMDb_URL', 'unknown', 'Action', 'Adventure', 'Animation', "Children's",
    'Comedy', 'Crime', 'Documentary', 'Drama', 'Fantasy', 'Film-Noir',
    'Horror', 'Musical', 'Mystery', 'Romance', 'Sci-Fi', 'Thriller',
    'War', 'Western'
]
movies = pd.read_csv("u.item", sep="|", names=items_cols, encoding="latin-1")

# ----------------------------------------
# Load user information
# ----------------------------------------
users_cols = ['user_id', 'age', 'gender', 'occupation', 'zip_code']
users = pd.read_csv("u.user", sep="|", names=users_cols)

# ----------------------------------------
# Load genres
# ----------------------------------------
genres = pd.read_csv("u.genre", sep="|", names=['genre', 'genre_id'], engine='python')

print("Ratings shape:", ratings.shape)
print("Movies shape:", movies.shape)
print("Users shape:", users.shape)
print("Genres shape:", genres.shape)

ratings.head(), movies.head(), users.head()

# ----------------------------------------
# Merge ratings with movies
# ----------------------------------------
ratings_movies = ratings.merge(movies, on="movie_id", how="left")

# ----------------------------------------
# Merge with users
# ----------------------------------------
full_df = ratings_movies.merge(users, on="user_id", how="left")

print("Merged shape:", full_df.shape)
full_df.head()

import matplotlib.pyplot as plt
import seaborn as sns

# ----------------------
# Rating Distribution
# ----------------------
plt.figure(figsize=(6,4))
sns.countplot(x=ratings["rating"])
plt.title("Rating Distribution")
plt.savefig("viz_rating_distribution.png")
plt.close()

# ----------------------
# Top Rated Movies (by count)
# ----------------------
top_movies = full_df.groupby("movie_title")["rating"].count().sort_values(ascending=False).head(10)
top_movies.to_csv("top_movies_by_count.csv")

# ----------------------
# Highest Rated Movies (min 100 ratings)
# ----------------------
movie_stats = full_df.groupby("movie_title").agg(
    avg_rating=("rating", "mean"),
    rating_count=("rating", "count")
)

best_movies = movie_stats[movie_stats["rating_count"] >= 100].sort_values("avg_rating", ascending=False).head(10)
best_movies.to_csv("best_movies_adjusted.csv")

# ----------------------
# Age distribution
# ----------------------
plt.figure(figsize=(6,4))
sns.histplot(users["age"], bins=20, kde=True)
plt.title("User Age Distribution")
plt.savefig("viz_age_distribution.png")
plt.close()

# ----------------------
# Gender rating analysis
# ----------------------
gender_avg = full_df.groupby("gender")["rating"].mean()
gender_avg.to_csv("gender_avg_rating.csv")

print("EDA saved: charts + CSV summaries")

# -------------------------------------------
# Build Genre Matrix for Content-based System
# -------------------------------------------

genre_cols = [
    'unknown', 'Action', 'Adventure', 'Animation', "Children's",
    'Comedy', 'Crime', 'Documentary', 'Drama', 'Fantasy', 'Film-Noir',
    'Horror', 'Musical', 'Mystery', 'Romance', 'Sci-Fi', 'Thriller',
    'War', 'Western'
]

# Genre matrix
genre_matrix = movies[genre_cols].values

# Normalize (not required but improves similarity slightly)
scaler = MinMaxScaler()
genre_matrix = scaler.fit_transform(genre_matrix)

# Compute similarity matrix
similarity_matrix = cosine_similarity(genre_matrix)

# Map movie titles to index
movie_indices = pd.Series(movies.index, index=movies['movie_title']).drop_duplicates()


# --------------------------------------------------
# Recommendation Function
# --------------------------------------------------
def recommend_movies(movie_title, top_n=10):
    if movie_title not in movie_indices:
        return f"❌ Movie '{movie_title}' not found!"

    idx = movie_indices[movie_title]
    sim_scores = list(enumerate(similarity_matrix[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:top_n+1]

    movie_ids = [i[0] for i in sim_scores]
    scores = [i[1] for i in sim_scores]

    results = movies.iloc[movie_ids][['movie_title', 'release_date']].copy()
    results["similarity"] = scores

    return results


# --------------------------------------------------
# Try a recommendation example
# --------------------------------------------------
print("Example Recommendations for 'Toy Story (1995)':")
print(recommend_movies("Toy Story (1995)"))

# -----------------------------
# Create User-Item Pivot Table
# -----------------------------
user_item_matrix = ratings.pivot(index='user_id', columns='movie_id', values='rating').fillna(0)

# -----------------------------
# Compute User Similarity (Cosine)
# -----------------------------

user_sim = cosine_similarity(user_item_matrix)
user_sim_df = pd.DataFrame(user_sim, index=user_item_matrix.index, columns=user_item_matrix.index)

# -----------------------------
# Recommendation Function
# -----------------------------
def recommend_for_user_pandas(user_id, top_n=10):
    if user_id not in user_sim_df.index:
        return f"❌ User {user_id} not found!"

    # Find similar users
    sim_scores = user_sim_df[user_id].sort_values(ascending=False)[1:]  # skip self
    top_users = sim_scores.index.tolist()

    # Weighted ratings from similar users
    similar_users_ratings = user_item_matrix.loc[top_users]
    weighted_ratings = similar_users_ratings.T.dot(sim_scores[top_users])
    
    # Remove already rated movies
    already_rated = user_item_matrix.loc[user_id]
    weighted_ratings = weighted_ratings[already_rated == 0]

    # Top N recommendations
    top_movies = weighted_ratings.sort_values(ascending=False).head(top_n)
    rec_df = movies[movies['movie_id'].isin(top_movies.index)][['movie_id','movie_title']].copy()
    rec_df['score'] = top_movies.values
    return rec_df

# -----------------------------
# Example
# -----------------------------
print("\nTop Recommendations for User 1 (Pandas CF):")
print(recommend_for_user_pandas(1))


# Save matrices
joblib.dump(similarity_matrix, "genre_similarity_matrix.pkl")
joblib.dump(user_sim_df, "user_similarity_matrix.pkl")

print(" Matrices saved. You can reload them anytime!")
