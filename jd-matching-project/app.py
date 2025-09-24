import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
from sklearn.metrics.pairwise import cosine_similarity

# Set page configuration
st.set_page_config(
    page_title="JD-Matching App",
    page_icon=":briefcase:",
    layout="wide",
)

# Function to load the models and data
def load_resources():
    try:
        # Get the base path of the script
        base_path = os.path.dirname(os.path.abspath(__file__))
        models_path = os.path.join(base_path, 'jd-matching-project', 'jd-matching-project', 'models')
        data_path = os.path.join(base_path, 'jd-matching-project', 'jd-matching-project', 'JD')

        # Load the TfidfVectorizer
        tfidf_path = os.path.join(models_path, 'tfidf_vectorizer.pkl')
        with open(tfidf_path, 'rb') as f:
            tfidf_vectorizer = pickle.load(f)
            
        # Load the similarity model (cosine similarity is used here, but this is a placeholder)
        similarity_model_path = os.path.join(models_path, 'similarity_model.pkl')
        with open(similarity_model_path, 'rb') as f:
            similarity_model = pickle.load(f)

        # Load the Job Descriptions from the JD folder (now from .txt files)
        job_descriptions = []
        for filename in os.listdir(data_path):
            if filename.endswith(".txt"):
                filepath = os.path.join(data_path, filename)
                with open(filepath, 'r') as f:
                    description = f.read()
                    # Create a job title from the filename
                    job_title = os.path.splitext(filename)[0].replace('_', ' ').replace('-', '/').title()
                    job_descriptions.append({'Job Title': job_title, 'Description': description})
        
        job_descriptions = pd.DataFrame(job_descriptions)

        if job_descriptions.empty:
            st.error("No job description files found in the 'JD' folder.")
            st.stop()

        return tfidf_vectorizer, similarity_model, job_descriptions
    except FileNotFoundError as e:
        st.error(f"Error: A required file was not found. Please ensure your 'models' and 'JD' folders are present and contain the necessary files. Missing file: {e.filename}")
        st.stop()
    except Exception as e:
        st.error(f"An unexpected error occurred while loading resources: {e}")
        st.stop()

# --- Main App Logic ---

st.title("JD-Matching Project")
st.subheader("Find the best-fit candidate for a Job Description!")

# Load the resources at the start
tfidf_vectorizer, similarity_model, job_descriptions = load_resources()

# User input for the job description
st.markdown("---")
st.write("### Enter a Job Description to find the best-matched candidate profiles.")
jd_input = st.text_area(
    "Job Description",
    height=250,
    placeholder="Paste the Job Description here..."
)

# Button to trigger the matching
if st.button("Find Best Match", use_container_width=True):
    if not jd_input:
        st.warning("Please enter a Job Description.")
    else:
        with st.spinner("Finding the best match..."):
            # Vectorize the input Job Description
            jd_vector = tfidf_vectorizer.transform([jd_input])

            # Vectorize all candidate descriptions
            candidate_vectors = tfidf_vectorizer.transform(job_descriptions['Description'])

            # Calculate cosine similarity between the input JD and all candidates
            similarities = cosine_similarity(jd_vector, candidate_vectors).flatten()

            # Find the index of the best match
            best_match_index = np.argmax(similarities)
            best_match_score = similarities[best_match_index] * 100 # Convert to percentage

            # Get the details of the best matched candidate
            best_match_details = job_descriptions.iloc[best_match_index]
            best_match_title = best_match_details['Job Title']

            st.success("Match found!")
            
            st.markdown("---")
            st.write(f"### Best Match Found: **{best_match_title}**")
            st.info(f"Similarity Score: **{best_match_score:.2f}%**")
            
            # Display the details of the best matched candidate
            st.markdown(f"**Job Title:** {best_match_details['Job Title']}")
            st.markdown(f"**Description:** {best_match_details['Description']}")
            st.markdown("---")


# --- File Upload Section (Optional for local testing) ---
st.markdown("---")
st.write("### Or, upload a custom dataset to use for matching.")
uploaded_file = st.file_uploader(
    "Upload a CSV file with 'Job Title' and 'Description' columns.",
    type=["csv"]
)

if uploaded_file:
    try:
        uploaded_data = pd.read_csv(uploaded_file)
        if 'Job Title' in uploaded_data.columns and 'Description' in uploaded_data.columns:
            st.success("File uploaded successfully!")
            st.write("Uploaded Data Preview:")
            st.dataframe(uploaded_data.head())

            # Now you can use this uploaded data in your matching logic
            # For example, you could append it to your existing data or replace it.
            # job_descriptions = pd.concat([job_descriptions, uploaded_data], ignore_index=True)
            # You would then need to re-vectorize the data if you added new descriptions.
        else:
            st.error("Invalid CSV file. Please ensure it has 'Job Title' and 'Description' columns.")
    except Exception as e:
        st.error(f"Error processing the file: {e}")
import streamlit as st

st.title('JD Matching App')
