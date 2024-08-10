import streamlit as st
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer

# Load and preprocess data
def load_and_preprocess_data(file_path):
    data = pd.read_csv(file_path)
    data = data.dropna()  # Handle missing values; replace with appropriate handling if needed
    # Convert the 'Question_Embedding' column from string to numpy array
    data['Question_Embedding'] = data['Question_Embedding'].apply(eval).apply(np.array)
    return data

# Calculate embeddings for user question
def calculate_user_embedding(question, model):
    return model.encode([question])[0]

# Find the best answer based on similarity
def find_best_answer(user_question, data, model, threshold=0.7):
    user_embedding = calculate_user_embedding(user_question, model)
    data['similarity'] = cosine_similarity([user_embedding], data['Question_Embedding'].tolist())[0]
    best_match = data.loc[data['similarity'].idxmax()]
    similarity_score = best_match['similarity']

    if similarity_score > threshold:
        return best_match['Answer'], similarity_score
    else:
        return "I apologize, but I don't have information on that topic yet. Could you please ask other questions?", 0

# Streamlit UI
def main():
    st.title("Health Question Answering System")

    # Load and preprocess the dataset
    data_file_path = 'qa_dataset_with_embeddings.csv'
    data = load_and_preprocess_data(data_file_path)

    # Load the embedding model
    model = SentenceTransformer('all-MiniLM-L6-v2')

    # User input for health question
    user_question = st.text_input("Ask your health question:")

    if st.button("Submit"):
        answer, similarity = find_best_answer(user_question, data, model)
        st.subheader("Answer:")
        st.write(answer)
        st.subheader("Similarity Score:")
        st.write(f"{similarity:.2f}")

    if st.button("Clear"):
        st.text_input("Ask your health question:", value="", key="new")

if __name__ == "__main__":
    main()
