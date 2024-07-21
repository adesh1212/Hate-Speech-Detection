import streamlit as st
import pickle
import re
import string
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

# Load the vectorizer and model
with open('../pythonProject/vectorizer.pkl', 'rb') as vectorizer_file:
    cv = pickle.load(vectorizer_file)
with open('model.pkl', 'rb') as model_file:
    clf_gini = pickle.load(model_file)

stopwords = stopwords.words('english')

# Define your text cleaning function
def clean(text):
    stemmer = PorterStemmer()

    # Convert text to lowercase
    text = str(text).lower()

    text = re.sub('\[.*?\]', '', text)

    # Remove URLs
    text = re.sub('https?://\S+|www\.\S+', '', text)

    # Remove HTML tags
    text = re.sub('<.*?>+', '', text)

    # Remove punctuation
    text = re.sub('[%s]' % re.escape(string.punctuation), '', text)

    # Remove newline characters
    text = re.sub('\n', '', text)

    # Remove digits or alphanumeric characters
    text = re.sub('\w*\d\w*', '', text)

    # Remove stopwords
    text = [word for word in text.split(' ') if word not in stopwords]
    text = ' '.join(text)

    # Apply stemming to words
    text = [stemmer.stem(word) for word in text.split(' ')]
    text = ' '.join(text)
    return text

# Streamlit app
st.title("Hate Speech Detection App")

# Input text from the user
input_text = st.text_area("Enter text to analyze")

if st.button("Predict"):
    # Preprocess the input text
    sample_processed = clean(input_text)

    # Vectorize the preprocessed input text
    sample_vector = cv.transform([sample_processed])

    # Predict the label for the input text
    sample_prediction = clf_gini.predict(sample_vector)
    prediction = sample_prediction[0]

    # Display the prediction result with color
    if prediction == "Normal":
        st.markdown(f"<h3 style='color:green;'>Prediction for the input text: {prediction}</h3>", unsafe_allow_html=True)
    else:
        st.markdown(f"<h3 style='color:red;'>Prediction for the input text: {prediction}</h3>", unsafe_allow_html=True)

# To run the app, use the following command in your terminal
# streamlit run app.py
