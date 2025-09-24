import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os

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

        # Try to find the files assuming a nested structure
        models_path = os.path.join(base_path, 'models')
        data_path = os.path.join(base_path, 'data')

        # Check if the nested path exists. If not, assume the files are at the root level.
        if not os.path.exists(models_path):
            models_path = os.path.join(base_path, 'jd-matching-project', 'jd-matching-project', 'models')
            data_path = os.path.join(base_path, 'jd-matching-project', 'jd-matching-project', 'data')

        # Load the TfidfVectorizer
        tfidf_path = os.path.join(models_path, 'tfidf_vectorizer.pkl')
        with open(tfidf_path, 'rb') as f:
            tfidf_vectorizer = pickle.load(f)
            
        # Load the similarity model
        similarity_model_path = os.path.join(models_path, 'similarity_model.pkl')
        with open(similarity_model_path, 'rb') as f:
            similarity_model = pickle.load(f)

        # Load the Job Descriptions from the data folder
        job_descriptions_path = os.path.join(data_path, 'jd_descriptions.csv')
        job_descriptions = pd.read_csv(job_descriptions_path)

        return tfidf_vectorizer, similarity_model, job_descriptions
    except FileNotFoundError as e:
        st.error(f"Error: A required file was not found. Please ensure your 'models' and 'data' folders are present and contain the necessary files. Missing file: {e.filename}")
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

# Get the list of available candidate profiles
candidate_profiles = job_descriptions['Job Title'].tolist()
selected_candidate = st.selectbox(
    "Select a candidate profile to use for demonstration:",
    options=[''] + candidate_profiles
)

# Button to trigger the matching
if st.button("Find Best Match", use_container_width=True):
    if not jd_input:
        st.warning("Please enter a Job Description.")
    elif not selected_candidate:
        st.warning("Please select a candidate profile.")
    else:
        with st.spinner("Finding the best match..."):
            # Placeholder for actual matching logic
            # In a real-world scenario, you would vectorize the jd_input and compare it
            # with vectorized candidate profiles using the similarity model.

            # For this example, we'll just show the selected candidate as a placeholder result.
            st.success("Match found!")
            
            st.markdown("---")
            st.write(f"### Best Match Found: **{selected_candidate}**")
            
            # Display the details of the selected candidate
            candidate_details = job_descriptions[job_descriptions['Job Title'] == selected_candidate].iloc[0]
            st.markdown(f"**Job Title:** {candidate_details['Job Title']}")
            st.markdown(f"**Description:** {candidate_details['Description']}")
            # In a real app, you would also display a similarity score
            st.markdown("---")
            st.info("Note: This is a placeholder result. In the full application, the model would perform a similarity calculation to find the actual best match.")

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
