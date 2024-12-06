import streamlit as st
import requests
import pandas as pd
from components.display import (
    display_topics,
    display_questions,
    display_analysis
)
from utils.helpers import process_upload

# Page config
st.set_page_config(
    page_title="Exam Paper Analyzer",
    page_icon="📚",
    layout="wide"
)

def main():
    st.title("Exam Paper Analyzer")
    
    # File upload
    uploaded_file = st.file_uploader("Upload PDF", type=['pdf'])
    
    if uploaded_file:
        with st.spinner('Analyzing paper...'):
            # Process the upload
            analysis_data = process_upload(uploaded_file)
            
            if analysis_data:
                # Create three columns
                col1, col2 = st.columns(2)
                
                with col1:
                    # Display topics
                    display_topics(analysis_data['topics'])
                    
                with col2:
                    # Display analysis results
                    display_analysis(analysis_data['results'])
                
                # Display questions in full width
                display_questions(analysis_data['questions'])
            else:
                st.error("Error processing the file. Please try again.")
    
    # Add sidebar info
    with st.sidebar:
        st.header("About")
        st.write("Upload exam papers to analyze common topics and questions.")
        
        st.header("Instructions")
        st.write("""
        1. Upload a PDF file of past exam papers
        2. Wait for analysis to complete
        3. View topic distribution and common questions
        """)

if __name__ == "__main__":
    main()