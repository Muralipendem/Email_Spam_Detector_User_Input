# Import Libraries
import pandas as pd
from flask import Flask, render_template, request
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

# Initialize the Flask app
app = Flask(__name__)

# Load and prepare the dataset
data = pd.read_csv('Email_spam_user\spam.csv', encoding='latin-1')
data = data[['v1', 'v2']]
data.columns = ['label', 'message']
data['label'] = data['label'].map({'ham': 0, 'spam': 1})

# Features (X) and target labels (y)
X = data['message']
y = data['label']

# Feature extraction using TF-IDF Vectorizer
tfidf = TfidfVectorizer(stop_words='english')
X_tfidf = tfidf.fit_transform(X)

# Train the Naive Bayes model
model = MultinomialNB()
model.fit(X_tfidf, y)

@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = None
    if request.method == 'POST':
        email_message = request.form['email_message']
        email_tfidf = tfidf.transform([email_message])  # Transform the input message
        prediction = model.predict(email_tfidf)[0]  # Predict spam or ham
        prediction = 'Spam' if prediction == 1 else 'Ham'
    return render_template('index.html', prediction=prediction)

if __name__ == '__main__':
    app.run(debug=True)
