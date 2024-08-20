import streamlit as st
from langchain_community.retrievers import PineconeHybridSearchRetriever
from pinecone import Pinecone, ServerlessSpec
from pinecone_text.sparse import BM25Encoder
from langchain_huggingface import HuggingFaceEmbeddings
import os
from dotenv import load_dotenv
import nltk

# Download the necessary nltk data
nltk.download('punkt_tab')

# Load environment variables
load_dotenv()
api_key = os.getenv("PINECONE_API_KEY")
index_name = "hybrid-search-langchain-pinecone"
pc = Pinecone(api_key=api_key)

# Create the index if it doesn't exist
if index_name not in pc.list_indexes().names():
    pc.create_index(
        name=index_name,
        dimension=384,  # dimensionality of dense model
        metric="dotproduct",  # sparse values supported only for dotproduct
        spec=ServerlessSpec(cloud="aws", region="us-east-1"),
    )

index = pc.Index(index_name)

# Set up embeddings and BM25 encoder
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
bm25_encoder = BM25Encoder().default()

# Define sample sentences
sentences = [
    "In 2023, I visited Paris",
    "In 2022, I visited New York",
    "In 2021, I visited New Orleans",
]

# Fit the BM25 encoder
bm25_encoder.fit(sentences)
bm25_encoder.dump("bm25_values.json")

# Load the BM25 encoder from the saved file
bm25_encoder = BM25Encoder().load("bm25_values.json")

# Set up the retriever
retriever = PineconeHybridSearchRetriever(embeddings=embeddings, sparse_encoder=bm25_encoder, index=index)

# Add texts to the index
retriever.add_texts(sentences)

# Streamlit app layout
st.markdown("""
    <style>
    .css-1d391kg {
        background-color: #1e1e1e;
        color: white;
    }
    .css-1v3fvcr {
        border-radius: 10px;
        background-color: #2c2c2c;
        color: #f0f0f0;
    }
    </style>
    """, unsafe_allow_html=True)

st.title("Chat-Mate..Hybrid Search with Langchain and Pinecone🔍")
st.write("Enter a query to search through the stored sentences.")

# User input
query = st.text_input("Enter your query:", placeholder="", key="query")

# Search and display the results
if st.button("Search"):
    if query.strip():
        try:
            result = retriever.invoke(query)
            if result:
                st.subheader("Search Results:")
                st.write(result[0].page_content)
            else:
                st.write("No matching results found.")
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")
    else:
        st.warning("Please enter a query to search.")
