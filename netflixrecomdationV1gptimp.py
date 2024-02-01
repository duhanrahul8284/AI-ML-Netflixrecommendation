import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import nltk
import re
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
import string  # Add this import statement

# Download stopwords
nltk.download('stopwords')

# Load data
data = pd.read_csv(r'C:\Users\freet\Downloads\netflixData.csv')

# Select relevant columns
data = data[["Title", "Description", "Content Type", "Genres"]].dropna()

# Define stopwords and stemmer
stopwords_set = set(stopwords.words('english'))
stemmer = SnowballStemmer("english")

# Cleaning function
def clean(text):
    text = str(text).lower()
    text = re.sub('\[.*?\]', '', text)
    text = re.sub('https?://\S+|www\.\S+', '', text)
    text = re.sub('<.*?>+', '', text)
    text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub('\n', ' ', text)
    text = re.sub('\w*\d\w*', '', text)
    words = [stemmer.stem(word) for word in text.split() if word not in stopwords_set]
    return " ".join(words)

# Apply cleaning to the "Genres" column
data["Genres"] = data["Genres"].apply(clean)

# Create a TF-IDF matrix
tfidf = TfidfVectorizer(stop_words="english")
tfidf_matrix = tfidf.fit_transform(data["Genres"])
similarity = cosine_similarity(tfidf_matrix)

# Create an index series
indices = pd.Series(data.index, index=data['Title']).drop_duplicates()

# Recommendation function
def netflix_recommendation(title, similarity_matrix=similarity):
    index = indices[title]
    similarity_scores = list(enumerate(similarity_matrix[index]))
    similarity_scores = sorted(similarity_scores, key=lambda x: x[1], reverse=True)
    similarity_scores = similarity_scores[1:11]  # Exclude the movie itself
    movie_indices = [i[0] for i in similarity_scores]
    return data.iloc[movie_indices][["Title", "Genres"]]

# Example usage
recommendations = netflix_recommendation("#Selfie")
print(recommendations)
