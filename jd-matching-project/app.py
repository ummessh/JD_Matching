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
        # Get the base path of the script, handling Streamlit's nested directory structure
        base_path = os.path.dirname(os.path.abspath(__file__))
        models_path = os.path.join(base_path, 'jd-matching-project', 'jd-matching-project', 'models')
        data_path = os.path.join(base_path, 'jd-matching-project', 'jd-matching-project', 'JD')

        # Load the TfidfVectorizer
        tfidf_path = os.path.join(models_path, 'tfidf_vectorizer.pkl')
        with open(tfidf_path, 'rb') as f:
            tfidf_vectorizer = pickle.load(f)
            
        # The similarity model is not a separate file, as cosine similarity is a built-in function.
        # This part of the code is now obsolete but kept for reference to the original request.
        similarity_model = None

        # Load the Job Descriptions from the JD folder (from .txt files)
        job_descriptions = {}
        for filename in os.listdir(data_path):
            if filename.endswith(".txt"):
                filepath = os.path.join(data_path, filename)
                with open(filepath, 'r') as f:
                    description = f.read()
                    job_title = os.path.splitext(filename)[0].replace('_', ' ').replace('-', ' / ').title()
                    job_descriptions[job_title] = description
        
        if not job_descriptions:
            st.error("No job description files found in the 'JD' folder.")
            st.stop()

        return tfidf_vectorizer, job_descriptions
    except FileNotFoundError as e:
        st.error(f"Error: A required file was not found. Please ensure your 'models' and 'JD' folders are present and contain the necessary files. Missing file: {e.filename}")
        st.stop()
    except Exception as e:
        st.error(f"An unexpected error occurred while loading resources: {e}")
        st.stop()

# --- Dummy Employee Database ---
# In a real-world scenario, this data would be loaded from a CSV, database, or API.
# For this example, we create a simple DataFrame.
def create_dummy_employee_data():
    data = {
        'Employee ID': [101, 102, 103, 104, 105, 106, 107, 108, 109, 110],
        'Name': ['Alice Johnson', 'Bob Smith', 'Charlie Brown', 'Diana Miller', 'Evan Davis', 'Fiona White', 'George Green', 'Hannah Black', 'Ian Taylor', 'Julia King'],
        'Skills': [
            'Python, TensorFlow, Keras, PyTorch, SQL, data modeling, deep learning, NLP',
            'Business analysis, stakeholder management, process improvement, agile, project management',
            'AWS, Azure, GCP, Docker, Kubernetes, DevOps, CI/CD, Terraform',
            'SQL, data visualization, Tableau, Power BI, Excel, statistics',
            'Machine learning, Python, Scikit-learn, model deployment, MLOps, scalability',
            'Digital marketing, social media campaigns, SEO, SEM, content strategy',
            'Android development, iOS development, React Native, Java, Swift, mobile UI/UX',
            'Network design, Cisco, Juniper, firewall, security protocols, troubleshooting',
            'UiPath, Blue Prism, Automation Anywhere, process automation, RPA',
            'Python, machine learning, data analysis, visualization, NLP'
        ],
        'Experience (Years)': [5, 8, 6, 7, 4, 10, 3, 9, 5, 6]
    }
    return pd.DataFrame(data)

# --- Main App Logic ---

st.title("JD-Matching Project")
st.subheader("Find the top 5 best-fit employees for a Job Description!")

# Load the resources at the start
tfidf_vectorizer, job_descriptions = load_resources()
employees_df = create_dummy_employee_data()

# User input: dropdown to select a JD
st.markdown("---")
selected_jd_title = st.selectbox(
    "Select a Job Description from the list:",
    list(job_descriptions.keys()),
    index=None,
    placeholder="Choose a Job Title..."
)

# Button to trigger the matching
if st.button("Find Top 5 Matches", use_container_width=True):
    if not selected_jd_title:
        st.warning("Please select a Job Description.")
    else:
        with st.spinner("Finding the best matches..."):
            # Get the description for the selected JD
            selected_jd_description = job_descriptions[selected_jd_title]

            # Vectorize the selected JD
            jd_vector = tfidf_vectorizer.transform([selected_jd_description])

            # Prepare employee data for vectorization
            employees_df['combined_text'] = employees_df['Skills'] + ' ' + employees_df['Experience (Years)'].astype(str) + ' years'
            
            # Vectorize all employee profiles
            employee_vectors = tfidf_vectorizer.transform(employees_df['combined_text'])

            # Calculate cosine similarity
            similarities = cosine_similarity(jd_vector, employee_vectors).flatten()

            # Add similarity scores to the DataFrame
            employees_df['Match Score'] = similarities * 100

            # Sort by match score and get the top 5
            top_5_matches = employees_df.sort_values(by='Match Score', ascending=False).head(5)

            st.success("Top 5 matches found!")
            
            st.markdown("---")
            st.write(f"### Top 5 Employees Matching: **{selected_jd_title}**")
            st.markdown("---")

            # Display results in a table
            display_df = top_5_matches[['Name', 'Skills', 'Match Score']]
            st.table(display_df.style.format({'Match Score': "{:.2f}%"}))
            
            # Display detailed view of the top match
            st.markdown("### Top Match Details")
            top_match = top_5_matches.iloc[0]
            st.info(f"**Name:** {top_match['Name']}\n\n**Skills:** {top_match['Skills']}\n\n**Experience:** {top_match['Experience (Years)']} years\n\n**Match Score:** {top_match['Match Score']:.2f}%")
