from langchain.vectorstores import FAISS
from langchain.llms import GooglePalm
from langchain.document_loaders.csv_loader import CSVLoader
from langchain.embeddings import HuggingFaceInstructEmbeddings
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Initialize LLM model with Google API key
language_model = GooglePalm(google_api_key=os.environ["GOOGLE_API_KEY"], temperature=0.1)

# Initialize embeddings with Hugging Face model
embeddings_model = HuggingFaceInstructEmbeddings(model_name="hkunlp/instructor-large")
vector_db_path = "faiss_index"

def build_vector_database():
    # Load data from CSV
    data_loader = CSVLoader(file_path='codebasics_faqs.csv', source_column="prompt")
    documents = data_loader.load()

    # Create and save FAISS vector database
    vector_db = FAISS.from_documents(documents=documents, embedding=embeddings_model)
    vector_db.save_local(vector_db_path)

def create_qa_chain():
    # Load vector database and create retriever
    vector_db = FAISS.load_local(vector_db_path, embeddings_model)
    retriever = vector_db.as_retriever(score_threshold=0.7)

    # Define prompt template
    prompt_format = """Given the following context and a question, generate an answer based on this context only.
    In the answer try to provide as much text as possible from "response" section in the source document context without making much changes.
    If the answer is not found in the context, kindly state "I don't know." Don't try to make up an answer.

    CONTEXT: {context}

    QUESTION: {question}"""

    prompt_template = PromptTemplate(
        template=prompt_format, input_variables=["context", "question"]
    )

    qa_chain = RetrievalQA.from_chain_type(
        llm=language_model,
        chain_type="stuff",
        retriever=retriever,
        input_key="query",
        return_source_documents=True,
        chain_type_kwargs={"prompt": prompt_template}
    )

    return qa_chain

if __name__ == "__main__":
    build_vector_database()
    chain = create_qa_chain()
    print(chain("Do you have javascript course?"))
