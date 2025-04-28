# Fake News Detection using Logistic Regression

This project implements a fake news detection model using **Logistic Regression** and text processing techniques. The model classifies news articles as either "Fake" or "Real" based on their content.

## Features
- **Text Preprocessing:** 
  - Converts text to lowercase.
  - Removes numbers and punctuation.
  - Removes stopwords and applies stemming for better feature extraction.
  
- **Vectorization:** 
  - Uses **TF-IDF** (Term Frequency - Inverse Document Frequency) vectorizer to convert text data into numerical features.
  
- **Model:** 
  - The model uses **Logistic Regression** for binary classification (Fake or Real).

- **Evaluation:** 
  - The model's accuracy is evaluated using the **test dataset**, and the accuracy score is printed.

- **User Input:** 
  - The model allows users to input their own news articles to predict whether the news is real or fake.

## How to Run

1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/fake-news-detection.git
   cd fake-news-detection

2. Install the required libraries:

pip install -r requirements.txt


3. Make sure you have the fake_or_real_news.csv dataset in your project directory.


4. Run the Python script:

python fake_news_detection.py


5. You will be prompted to enter a news article. The model will predict if the news is Fake or Real and print the result.



Model Accuracy

The model's accuracy on the test dataset is evaluated and displayed after training.

The accuracy score may vary based on the dataset and preprocessing techniques used.


Files

fake_news_detection.py: Main Python script for training and prediction.

fake_or_real_news.csv: Dataset of news articles (make sure it's placed in the same directory as the script).

requirements.txt: List of Python dependencies required to run the project.


Requirements

Python 3.x

scikit-learn

pandas

numpy

nltk

re

string
