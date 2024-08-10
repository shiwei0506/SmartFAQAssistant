import streamlit as st
import openai
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer

# Set up the OpenAI API key
openai.api_key = st.secrets["mykey"]

# Load and preprocess data
def load_and_preprocess_data(file_path):
    data = pd.read_csv(file_path)
    data = data.dropna()  # Handle missing values; replace with appropriate handling if needed
    # Convert the 'Question_Embedding' column from string to numpy array
    data['Question_Embedding'] = data['Question_Embedding'].apply(eval).apply(np.array)
    return data

# Function to generate a detailed response using GPT-3.5
def generate_gpt3_answer(user_question):
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a helpful health assistant."},
            {"role": "user", "content": user_question}
        ]
    )
    return response['choices'][0]['message']['content']

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
        # If no close match is found, use GPT-3.5 to generate a detailed response
        gpt3_answer = generate_gpt3_answer(user_question)
        return gpt3_answer, similarity_score

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

    # Placeholder for additional features like user ratings, FAQs, etc.

if __name__ == "__main__":
    main()
