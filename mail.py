import matplotlib.pyplot as plt
import pickle
import streamlit as st
from nltk.corpus import stopwords
import nltk
from nltk.stem.porter import PorterStemmer
import string

ps=PorterStemmer()
nltk.download('stopwords')

# Download the punkt tokenizer data
nltk.download('punkt')

# Function to preprocess text
def transform_text(text):
    # Convert to lowercase
    text = text.lower()
    # Tokenize the text
    text = nltk.word_tokenize(text)
    # Remove non-alphanumeric characters
    y = [i for i in text if i.isalnum()]
    # Remove stopwords and punctuation
    y = [i for i in y if i not in stopwords.words('english')]
    # Stem the words
    y = [ps.stem(i) for i in y]
    return " ".join(y)

tfidf=pickle.load(open('vectorizer.pkl', "rb"))
model=pickle.load(open('model.pkl', "rb"))

uploaded_file = st.file_uploader("Upload a text file", type=["txt"])


st.title(" Spam Detector")
input_sms=st.text_area("Enter your message")

result = None


if st.button('predict'):
    if uploaded_file is not None:
        input_sms = uploaded_file.read().decode("utf-8")
        transformed_sms = transform_text(input_sms)

        st.text_area("File Content", input_sms, height=200)

        # Vectorize and predict
        vector_input = tfidf.transform([transformed_sms])
        result = model.predict(vector_input)[0]

    elif input_sms:
        # Preprocess the input text
        transformed_sms = transform_text(input_sms)

        # Vectorize the input text
        vector_input = tfidf.transform([transformed_sms])

        # Predict the label
        result = model.predict(vector_input)[0]

        # Display the result
    if uploaded_file or input_sms:
        # Display the result
        if result == 1:
            st.header("Spam")
        else:
            st.header("Not Spam")





