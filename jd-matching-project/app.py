import streamlit as st
import os
import pandas as pd
from src.preprocessing import clean_text
from src.embeddings import build_vectorizer, vectorize_texts
from src.matching import get_top_n_matches

# Paths
JD_FOLDER = "JD"
EMPLOYEE_DB = "Data/employees.csv"

# Load employee database
@st.cache_data
def load_employee_db():
    try:
        return pd.read_csv(EMPLOYEE_DB)
    except FileNotFoundError:
        st.error(f"Employee database file not found: {EMPLOYEE_DB}")
        st.info("Please ensure the employees.csv file exists in the Data/ directory.")
        return pd.DataFrame()

# Load available JDs from folder
@st.cache_data
def load_jds():
    try:
        if not os.path.exists(JD_FOLDER):
            st.error(f"JD folder not found: {JD_FOLDER}")
            return {}
        
        jd_files = [f for f in os.listdir(JD_FOLDER) if f.endswith(".txt")]
        if not jd_files:
            st.warning(f"No .txt files found in {JD_FOLDER} directory.")
            return {}
        
        jds = {}
        for file in jd_files:
            with open(os.path.join(JD_FOLDER, file), "r", encoding="utf-8") as f:
                jds[file] = f.read()
        return jds
    except Exception as e:
        st.error(f"Error loading JD files: {str(e)}")
        return {}

def main():
    st.title("🔍 JD Matching App")
    st.write("Compare a job description against the employee database and get the top 5 matches.")
    
    # Load data
    employees = load_employee_db()
    jds = load_jds()
    
    # Check if data loaded successfully
    if employees.empty:
        st.stop()
    
    if not jds:
        st.stop()
    
    # Check if required column exists
    if "current_jd" not in employees.columns:
        st.error("Employee database must contain a 'current_jd' column.")
        st.stop()
    
    # Dropdown for JD selection
    jd_choice = st.selectbox("Select a Job Description to compare:", list(jds.keys()))
    
    if jd_choice:
        input_jd = jds[jd_choice]
        
        # Preprocess texts
        employee_jds = employees["current_jd"].apply(clean_text).tolist()
        input_jd_clean = clean_text(input_jd)
        
        # Build TF-IDF vectorizer
        vectorizer = build_vectorizer(employee_jds + [input_jd_clean])
        
        # Vectorize
        employee_vecs = vectorize_texts(employee_jds, vectorizer)
        input_vec = vectorize_texts([input_jd_clean], vectorizer)
        
        # Get top 5 matches
        top_matches = get_top_n_matches(input_vec, employee_vecs, employees, n=5)
        
        # Show results
        st.subheader("🏆 Top 5 Matching Employees")
        st.dataframe(top_matches.reset_index(drop=True))

if __name__ == "__main__":
    main()
