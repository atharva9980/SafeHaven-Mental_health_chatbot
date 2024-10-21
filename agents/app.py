import streamlit as st
import os
import langchain
from langchain_groq import ChatGroq
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.embeddings import OllamaEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain_community.vectorstores import FAISS
import time
import requests
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from langchain.schema import Document


load_dotenv()

## Load the Groq API key
groq_api_key = os.environ['GROQ_API_KEY']

def fetch_web_content(url):
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3'}
    try:
        response = requests.get(url, headers=headers)
        if response.status_code == 200:
            soup = BeautifulSoup(response.content, "html.parser")
            page_text = soup.get_text()
            return page_text.strip()
        else:
            st.error(f"Failed to retrieve content from {url}, status code: {response.status_code}")
            return None
    except Exception as e:
        st.error(f"Error fetching content from {url}: {str(e)}")
        return None


if "vector" not in st.session_state:
    st.session_state.embeddings = OllamaEmbeddings(model='all-minilm')

    # Load multiple PDFs
    pdf_paths = ["C:/Users/athar/OneDrive/Documents/mental_health_chatbot/agents/Depression__The_Importance_Of_Mental_Health_Awareness.pdf",
                 
                 "C:/Users/athar/OneDrive/Documents/mental_health_chatbot/agents/how to stay happy.pdf",
                 "C:/Users/athar/OneDrive/Documents/mental_health_chatbot/agents/overthinking.pdf",
                 "C:/Users/athar/OneDrive/Documents/mental_health_chatbot/agents/Mindfulness_for_Accepting_Depression_&_Anxiety_Symptoms.pdf"
    
    ]
    st.session_state.docs = []

    for pdf_path in pdf_paths:
        loader = PyPDFLoader(pdf_path)
        st.session_state.docs.extend(loader.load())

    # Fetch content from multiple websites
    websites = [
        "https://www.nimh.nih.gov/health/topics/depression",
        "https://www.nhs.uk/mental-health/self-help/tips-and-support/how-to-be-happier/",
        "https://www.nimh.nih.gov/health/topics/suicide-prevention#:~:text=Suicide%20is%20a%20major%20public,help%20can%20help%20save%20lives",
        "https://www.nimh.nih.gov/health/topics/anxiety-disorders"
        
    ]

    for url in websites:
        content = fetch_web_content(url)
        if content:
            st.session_state.docs.append(Document(page_content=content, metadata={"source": url}))

    # Split and vectorize documents
    st.session_state.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    st.session_state.final_documents = st.session_state.text_splitter.split_documents(st.session_state.docs)
    st.session_state.vectors = FAISS.from_documents(st.session_state.final_documents, st.session_state.embeddings)

st.title("SafeHaven")
st.sidebar.header("Configuration")

llm = ChatGroq(groq_api_key=groq_api_key, model_name="llama3-8b-8192")

prompt = ChatPromptTemplate.from_template(
"""
Answer the questions based on the provided context.
Please provide the most accurate response based on the question.
<context>
{context}
<context>
Questions: {input}
"""
)

document_chain = create_stuff_documents_chain(llm, prompt)
retriever = st.session_state.vectors.as_retriever()
retrieval_chain = create_retrieval_chain(retriever, document_chain)

# Load the SentenceTransformer model for semantic similarity
semantic_model = SentenceTransformer('all-MiniLM-L6-v2')

def compute_semantic_similarity(query, documents):
    query_embedding = semantic_model.encode([query])
    doc_embeddings = semantic_model.encode([doc.page_content for doc in documents])
    similarities = cosine_similarity(query_embedding, doc_embeddings)[0]
    return similarities

prompt = st.text_input("Input your prompt here")

if prompt:
    start = time.process_time()
    response = retrieval_chain.invoke({"input": prompt})
    context_docs = response["context"]

    # Compute semantic similarity between the input query and retrieved documents
    similarities = compute_semantic_similarity(prompt, context_docs)

    # Combine documents with their similarity scores and sort by similarity
    sorted_docs = sorted(zip(context_docs, similarities), key=lambda x: x[1], reverse=True)

    # Display response and the most relevant documents
    st.write(response['answer'])

    with st.expander("Document Similarity Search"):
        for i, (doc, sim) in enumerate(sorted_docs):
            st.write(f"Document {i+1} (Similarity: {sim:.2f}):")
            st.write(doc.page_content)
            st.write("--------------------------------")
