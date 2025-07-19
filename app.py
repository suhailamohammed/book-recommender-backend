import numpy as np
import pandas as pd
from fastapi import FastAPI
from sklearn.metrics.pairwise import cosine_similarity
from fastapi.responses import JSONResponse

app = FastAPI()

#loading datasets and matrix
file_path_dataset = 'global_book_dataset.csv'
file_path_matrix  = 'combined_features.npy'

df_global         = pd.read_csv(file_path_dataset)
combined_features = np.load(file_path_matrix)

# Precompute titles set of user's books (for filtering)
my_shelves    = ['read', 'to-read', 'currently-reading', 'best-of-best']
my_books_df   = df_global[df_global['Exclusive Shelf'].isin(my_shelves)]
my_titles_set = set(my_books_df['title'].str.lower().str.strip())

@app.get('/recommend')
def recommend_books(top_n: int = 20, shuffle: bool = False):
    my_indices = my_books_df.index.to_list()
    
    # Compute similarity of user's books to all books
    user_sim = cosine_similarity(combined_features[my_indices], combined_features)
    mean_sim = user_sim.mean(axis=0)
    
    # Exclude already interacted books
    mean_sim[my_indices] = 0
    
    # Sort by similarity descending
    recommended_indices = mean_sim.argsort()[::-1]
    
    # Optionally shuffle a bit for variety
    if shuffle:
        recommended_indices = np.random.permutation(recommended_indices)
    
    final_recommendations = []
    for idx in recommended_indices:
        title = df_global.at[idx, 'title'].lower()
        if title not in my_titles_set:
            final_recommendations.append(idx)
        if len(final_recommendations) >= top_n:
            break
    
    rec_books = df_global.loc[final_recommendations][['title', 'authors', 'Average Rating', 'categories', 'Exclusive Shelf']]

     # Clean data and convert to list of dictionaries
    rec_books_clean = rec_books.replace([np.inf, -np.inf], np.nan).fillna("").to_dict(orient="records")

    return JSONResponse(content={"recommendations": rec_books_clean})