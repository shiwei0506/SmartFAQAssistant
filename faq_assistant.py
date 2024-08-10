import streamlit as st
import openai
import pandas as pd
import openai

openai.api_key =  st.secrets["mykey"]

# Load and preprocess data
def load_and_preprocess_data(file_path):
    data = pd.read_csv(file_path)
    data = data.dropna()  # Handle missing values; replace with appropriate handling if needed
    # Perform any additional preprocessing steps
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

# Streamlit UI
def main():
    st.title("Health Question Answering System")

    # Load and preprocess the dataset
    data_file_path = 'qa_dataset_with_embeddings.csv'
    data = load_and_preprocess_data(data_file_path)

    # User input for health question
    user_question = st.text_input("Ask your health question:")

    if st.button("Submit"):
        # Generate answer using GPT-3.5
        gpt3_answer = generate_gpt3_answer(user_question)
        st.subheader("Answer:")
        st.write(gpt3_answer)

    if st.button("Clear"):
        st.text_input("Ask your health question:", value="", key="new")

    # Placeholder for additional features like user ratings, FAQs, etc.

if __name__ == "__main__":
    main()
