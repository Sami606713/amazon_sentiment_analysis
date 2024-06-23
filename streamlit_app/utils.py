import pickle as pkl
import string
import nltk
from nltk.stem import PorterStemmer ,WordNetLemmatizer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
# Download NLTK data
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# Load the model
def load_model():
    with open("streamlit_app/models/Sentiment.pkl" ,"rb") as f:
        model=pkl.load(f)
    return model

# Define the process fun that can process the text
def processor(text):
    """
    we can perform text preprocessing on the input text:
    1. Convert text to lowercase.
    2. Remove punctuation.
    3. Tokenize the text into words.
    4. Remove English stopwords.
    5. Stem and lemmatize the remaining words.
    After completing these step we can join the text and return them
    """
        
    text=text.lower()
    
    # remove pouncation
    translator=str.maketrans('', '', string.punctuation)
    text=text.translate(translator)
    
    
    # Tokenize the word
    token=word_tokenize(text)
    
    # Remove Stop Words
    stop_words=set(stopwords.words('english'))
    update_text=[word for word in token if word.lower() not in stop_words]
    
    # Stem the word
    stem=PorterStemmer()
    stem_text=[stem.stem(word) for word in  update_text]
    
    # Lemitize them
    lem=WordNetLemmatizer()
    final_text=[lem.lemmatize(word) for word in stem_text]
    
    return " ".join(final_text)