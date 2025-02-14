import streamlit as st
import nltk
from transformers import pipeline
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from transformers import AutoTokenizer, AutoModelForCausalLM

tokenizer = AutoTokenizer.from_pretrained("m42-health/Llama3-Med42-70B")
model = AutoModelForCausalLM.from_pretrained("m42-health/Llama3-Med42-70B")



def healthcare_chatbot(user_input):
    if "remedy" in user_input:
        return "Sorry, I can not provide this information. If you have any health concerns, please consult a doctor."
    elif "appointment" in user_input:
        return "Would you like to schedule appointment with the doctor?"
    elif "medication" in user_input:
        return "It's important to take prescribed medication regularly. If you have concerns, please consult your doctor."
    else:
        inputs = tokenizer(user_input, return_tensors="pt")
        output = model.generate(**input, max_length = 500, num_return_sequences=1)
        response = tokenizer.decode(output[0], skip_special_tokens=True)
        return response[0]['generated_text']

def main():
    st.title("Healthcare Assistant Chatbot")
    user_input = st.text_input("How can I help you today?")
    if st.button("Submit"):
        if user_input:
            st.write("User : ", user_input)
            with st.spinner("Processing your query, Please wait..."):
                response = healthcare_chatbot(user_input)
            st.write("Healthcare Assistant : ", response)
        else:
            st.write("Please enter a message to get a response")
            
if __name__ == "__main__":
    main()
