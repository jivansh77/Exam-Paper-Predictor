import streamlit as st
import plotly.express as px
import pandas as pd

def display_topics(topics):
    st.subheader("Topic Distribution")
    
    # Create DataFrame for visualization
    df = pd.DataFrame(topics)
    
    # Create bar chart
    fig = px.bar(df, x='name', y='frequency', 
                 title='Topic Frequencies',
                 labels={'name': 'Topic', 'frequency': 'Frequency (%)'},
                 color='frequency')
    
    st.plotly_chart(fig)

def display_questions(questions):
    st.subheader("Common Questions")
    
    # Filter controls
    col1, col2 = st.columns(2)
    with col1:
        question_type = st.selectbox("Filter by Type", 
                                   ['All', 'Theory', 'Numerical'])
    with col2:
        min_relevance = st.slider("Minimum Relevance", 0, 100, 50)
    
    # Filter questions
    filtered_questions = [
        q for q in questions 
        if (question_type == 'All' or q['type'].lower() == question_type.lower()) 
        and q['relevance'] >= min_relevance
    ]
    
    # Display questions
    for q in filtered_questions:
        with st.expander(f"{q['text'][:100]}..."):
            st.write(f"**Full Question:** {q['text']}")
            st.write(f"**Type:** {q['type']}")
            st.write(f"**Topic:** {q['topic']}")
            st.write(f"**Relevance:** {q['relevance']}%")

def display_analysis(results):
    st.subheader("Analysis Summary")
    
    # Display metrics
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Questions", results['totalQuestions'])
    col2.metric("Theory Questions", results['theoryCount'])
    col3.metric("Numerical Questions", results['numericalCount'])
    
    # Display patterns
    if results.get('patterns'):
        st.subheader("Identified Patterns")
        for pattern in results['patterns']:
            st.info(pattern)