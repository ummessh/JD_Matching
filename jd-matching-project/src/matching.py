import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

def get_top_n_matches(input_vec, employee_vecs, employees_df, n=5):
    """
    Compare input JD vector with employee JD vectors.
    Returns top n employees with highest similarity.
    """
    # Compute cosine similarity
    sims = cosine_similarity(input_vec, employee_vecs).flatten()
    
    # Add similarity scores to employee dataframe
    employees_df = employees_df.copy()
    employees_df["similarity"] = sims
    
    # Sort by similarity
    top_matches = employees_df.sort_values(by="similarity", ascending=False).head(n)
    
    return top_matches[["employee_id", "name", "current_jd", "similarity"]]

