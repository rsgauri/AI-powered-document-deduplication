import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import AffinityPropagation
from sklearn.metrics.pairwise import cosine_similarity
from docx import Document
from nltk.tokenize import word_tokenize
import re
import nltk
import os
import pandas as pd

# Download NLTK data needed for text processing
nltk.download('stopwords')
nltk.download('punkt')

# Function to read text from a .docx file
def read_docx(file_path):
    doc = Document(file_path)
    paragraphs = [paragraph.text for paragraph in doc.paragraphs]
    text = ' '.join(paragraphs)
    return text

# Function to remove common stopwords from text
def clean_stopwords(text: str) -> str:
    stopwords_list = ["a", "an", "and", "at", "but", "how", "in", "is", "on", "or", "the", "to", "what", "will"]
    tokens = word_tokenize(text)
    clean_tokens = [t for t in tokens if t.lower() not in stopwords_list]
    return " ".join(clean_tokens)

# Function to clean up text by removing URLs, emails, and special characters
def clean_text(text):
    text = re.sub(r'http\S+', '', text)  # Remove URLs
    text = re.sub(r'\S+@\S+', '', text)  # Remove emails
    text = re.sub(r'[^a-zA-Z\s]', '', text)  # Remove non-alphabetic characters
    text = ' '.join(text.split())  # Remove extra spaces
    return text

# Function to cluster documents using Affinity Propagation algorithm
def cluster_documents(documents):
    vectorizer = TfidfVectorizer(stop_words='english')
    X = vectorizer.fit_transform(documents)  # Convert text to TF-IDF matrix
    model = AffinityPropagation()
    model.fit(X)  # Perform clustering
    labels = model.labels_  # Get the labels assigned to each document

    # Organize documents by cluster
    cluster_docs = {}
    for i in set(labels):
        cluster_docs[i] = [documents[j] for j in range(len(labels)) if labels[j] == i]
    return cluster_docs

# Function to compute cosine similarity
def compute_cosine_similarity(doc1, doc2):
    vectorizer = TfidfVectorizer(stop_words='english')
    tfidf_matrix = vectorizer.fit_transform([doc1, doc2])
    similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])
    return similarity[0][0]

# Streamlit app
st.title("Document Clustering and Cosine Similarity")

uploaded_files = st.file_uploader("Upload .docx files for clustering", type="docx", accept_multiple_files=True)
consolidated_files = st.file_uploader("Upload consolidated .docx files", type="docx", accept_multiple_files=True)
process_button = st.button("Submit")

if process_button:
    if not uploaded_files or not consolidated_files:
        st.warning("Please upload both document files.")
    else:
        documents = [read_docx(file) for file in uploaded_files]
        st.write("Clustering documents...")
        cluster_docs = cluster_documents(documents)

        st.write("Calculating Cosine Similarity...")
        results = []

        for consolidated_file in consolidated_files:
            consolidated_text = read_docx(consolidated_file)
            consolidated_name = os.path.splitext(consolidated_file.name)[0]
            matched_cluster = None
            max_similarity = 0

            for cluster_number, docs in cluster_docs.items():
                clustered_text = " ".join(docs)
                similarity_score = compute_cosine_similarity(clustered_text, consolidated_text)
                if similarity_score > max_similarity:
                    max_similarity = similarity_score
                    matched_cluster = cluster_number

            results.append({
                "Consolidated Document": consolidated_name,
                "Matched Cluster": matched_cluster,
                "Cosine Similarity": max_similarity
            })

        # Convert results to a DataFrame and display as a table
        df = pd.DataFrame(results)
        st.write(df)
