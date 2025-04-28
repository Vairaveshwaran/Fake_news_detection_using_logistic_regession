import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import nltk
import re
import string

nltk.download('stopwords')

# Load dataset
df = pd.read_csv("fake_or_real_news.csv")

# Text preprocessing function
def clean_text(text):
    text = text.lower() # convert to lowercase
    text = re.sub(r'\d+', '', text) # remove numbers
    text = text.translate(str.maketrans('', '', string.punctuation)) # remove punctuation
    text = text.strip() # remove leading/trailing whitespaces
    words = text.split()
    # Remove stopwords
    words = [word for word in words if word not in stopwords.words('english')]
    # Optional: stemming
    stemmer = PorterStemmer()
    words = [stemmer.stem(word) for word in words]
    return " ".join(words)

# Apply cleaning to text column
df['cleaned_text'] = df['text'].apply(clean_text)

# Split data into features (X) and labels (y)
X = df['cleaned_text'] # news content
y = df['label'] # real (1) or fake (0)

# Vectorization with TF-IDF (more effective than CountVectorizer)
vectorizer = TfidfVectorizer(ngram_range=(1, 2), max_features=5000) # TF-IDF with bigrams
X_vectorized = vectorizer.fit_transform(X)

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_vectorized, y, test_size=0.2, random_state=42)

# Train logistic regression model
model = LogisticRegression(max_iter=1000, C=1.0, solver='liblinear')
model.fit(X_train, y_train)

# Evaluate model accuracy on the test data
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

# Print the model accuracy score
print(f"Model Accuracy: {accuracy * 100:.2f}%")

# Function to predict user input
def predict_news(news_input):
    cleaned_input = clean_text(news_input) # Preprocess the input
    input_vectorized = vectorizer.transform([cleaned_input]) # Convert to numeric features using the same vectorizer
    prediction = model.predict(input_vectorized) # Predict the label
    return "Real News" if prediction == 1 else "Fake News" # Return corresponding result

# Get user input and predict
user_input = input("Enter the news article: ") # Take news input from user
result = predict_news(user_input) # Predict whether it's real or fake
print(result)