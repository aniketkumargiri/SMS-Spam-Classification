import streamlit as st
import pickle
import string
from nltk.corpus import stopwords
import nltk
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

ps = PorterStemmer()

# Step 1


def transform_text(text):
    # lower casing
    text = text.lower()

    # tokenization
    text = nltk.word_tokenize(text)

    # removing special characters
    y = []
    for i in text:
        if i.isalnum():
            y.append(i)

    text = y[:]
    y.clear()

    # removing stopwords and punctuation
    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)

    text = y[:]
    y.clear()

    # stemming
    for i in text:
        y.append(ps.stem(i))

    return " ".join(y)


tfidf = pickle.load(open('vectorizer.pkl', 'rb'))
model = pickle.load(open('model.pkl', 'rb'))

# Set page title and favicon
st.set_page_config(
    page_title="Email/SMS Spam Classification",
    page_icon="📧"
)

# Set page background color
st.markdown(
    """
    <style>
    body {
        background-color: #f5f7f9;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Set page title
st.title("Email/SMS Spam Classification")

# Navigation bar
nav_selection = st.sidebar.radio("Navigation", ["Home", "Contact", "Help"])

if nav_selection == "Home":
    st.header("Home")

    # Text input for user message
    input_sms = st.text_area("Enter the message")

    # Button to predict spam/ham
    if st.button('Predict'):
        # Preprocess input
        transformed_sms = transform_text(input_sms)

        # Vectorize input
        vector_input = tfidf.transform([transformed_sms])

        # Make prediction
        result = model.predict(vector_input)[0]

        # Display result
        if result == 1:
            st.success("This is a spam message")
        else:
            st.success("This is not a spam message")

elif nav_selection == "Contact":
    st.header("Contact")

    # Contact information
    st.subheader("Email: aniket.kumar.giri2707@gmail.com")
    st.subheader("Phone: +91 9123997300")

elif nav_selection == "Help":
    st.header("Help")

    # Help information
    st.write("If you need any assistance, please reach out to us.")

# Footer
st.markdown("---")
st.write("Made with ❤️ by Aniket Giri")
