import streamlit as st
import nltk
from transformers import pipeline
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

chatbot = pipeline("text-generation", model="microsoft/DialoGPT-medium")

def healthcare_chatbot(user_input):
    response = chatbot(user_input, max_length = 500, num_return_sequences=1)
    return response[0]['generated_text']

def main():
    st.title("Healthcare Assistant Chatbot")
    user_input = st.text_input("How can I help you today?")
    if st.button("Submit"):
        if user_input:
            st.write("User : ", user_input)
            # with st.spinner("Processing your query, Please wait..."):
            #     response = healthcare_chatbot(user_input)
            response = '''Sorry, I can not provide this information. If you have any health concerns, please consult a doctor.'''
            st.write("Healthcare Assistant : ", response)
        else:
            st.write("Please enter a message to get a response")
            
if __name__ == "__main__":
    main()