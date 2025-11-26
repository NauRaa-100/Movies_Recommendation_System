
import numpy as np
import pandas as pd
import joblib
import gradio as gr
from analysis import movie_indices,movies,user_sim_df , recommend_movies,recommend_for_user_pandas
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

# Reload matrices
similarity_matrix = joblib.load("genre_similarity_matrix.pkl")
user_sim_df = joblib.load("user_similarity_matrix.pkl")

def recommend_movies_cb(movie_title, top_n=10):
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
movie_list = movies['movie_title'].tolist()

def dashboard(movie_input, user_input):
    results_cb = recommend_movies(movie_input) if movie_input else None
    results_cf = recommend_for_user_pandas(user_input) if user_input else None
    return results_cb, results_cf

# ---------------------------
# Plot 1: Rating Distribution
# ---------------------------
def plot_rating_distribution():
    
    ratings = pd.read_csv("u.data", sep="\t", names=["user_id","movie_id","rating","timestamp"])
    fig = px.histogram(ratings, x="rating", nbins=5, title="Rating Distribution")
    return fig

# ---------------------------
# Plot 2: Genre Popularity
# ---------------------------
def plot_genre_popularity():
    genre_cols = [
        'unknown','Action','Adventure','Animation',"Children's",'Comedy','Crime',
        'Documentary','Drama','Fantasy','Film-Noir','Horror','Musical',
        'Mystery','Romance','Sci-Fi','Thriller','War','Western'
    ]
    genre_sums = movies[genre_cols].sum().sort_values(ascending=False)
    fig = px.bar(genre_sums, title="Genre Popularity", labels={'value': 'Count', 'index': 'Genre'})
    return fig

# ---------------------------
# Plot 3: User–Movie Interaction Heatmap (small sample)
# ---------------------------
def plot_heatmap():
    ratings = pd.read_csv("u.data", sep="\t", names=["user_id","movie_id","rating","timestamp"])
    pivot = ratings.pivot_table(index="user_id", columns="movie_id", values="rating").fillna(0)
    pivot_sample = pivot.iloc[:30, :30]  # Small sample
    
    fig = px.imshow(pivot_sample,
                    aspect="auto",
                    title="User-Movie Rating Heatmap (Sample 30x30)")
    return fig


# ---------------------------
# Dashboard Function
# ---------------------------
def dashboard(movie_input, user_input):
    results_cb = recommend_movies(movie_input) if movie_input else None
    results_cf = recommend_for_user_pandas(user_input) if user_input else None
    return (
        results_cb,
        results_cf,
        plot_rating_distribution(),
        plot_genre_popularity(),
        plot_heatmap()
    )

# ---------------------------
# Gradio UI
# ---------------------------
with gr.Blocks() as demo:

    gr.Markdown("## 🎬 Movie Recommendation System")

    with gr.Row():
        with gr.Column():
            movie_input = gr.Dropdown(choices=movie_list, label="Content-Based: Select a Movie")
            cb_output = gr.Dataframe(headers=["movie_title","release_date","similarity"])
            
        with gr.Column():
            user_input = gr.Number(
                label="Collaborative Filtering: Enter User ID (1–943)"
            )

            cf_output = gr.Dataframe(
                headers=["movie_id", "movie_title", "score"],
            )
  

    # Charts Section
    with gr.Row():
        rating_chart = gr.Plot(label="Rating Distribution")
        genre_chart = gr.Plot(label="Genre Popularity")
        heatmap_chart = gr.Plot(label="User–Movie Heatmap Sample")

    gr.Button("Generate Recommendations & Visuals").click(
        dashboard,
        inputs=[movie_input, user_input],
        outputs=[cb_output, cf_output, rating_chart, genre_chart, heatmap_chart]
    )

demo.launch(share=True)
