import streamlit as st
from code import create_qa_chain, build_vector_database

# Set up the title of the app
st.title("Codebasics Q&A 🌱")

# Button to create the knowledge base
if st.button("Create Knowledgebase"):
    build_vector_database()
    st.success("Knowledgebase created successfully!")

# Input for the question
question = st.text_input("Question:")

if question:
    # Generate the Q&A chain
    qa_chain = create_qa_chain()
    
    # Get the response
    response = qa_chain(question)
    
    # Display the answer
    st.header("Answer")
    st.write(response["result"])
