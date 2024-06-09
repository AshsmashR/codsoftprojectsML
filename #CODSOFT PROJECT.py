#CODSOFT PROJECT 
#ML TASK 1
#MOVIE GENRE CLASSIFICATION

import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sns

#uploading my movie genre dataset 
data_path = r"E:\Genre Classification Dataset\train_data.txt"
training_data = pd.read_csv(data_path, sep=':::', names=['Title', 'Genre', 'Description'], engine='python')
training_data.head(6)
training_data.info()
training_data.describe()
X = training_data
X.shape
training_data.drop('Lifestyle', axis=1, inplace=True, errors='ignore')
print(training_data.columns)

#CHECKING THE NAN/NULL VALUES 
training_data.isnull().sum()
class_distribution = training_data['Genre'].value_counts()
print("Class Distribution:")
print(class_distribution)
imbalance_ratio = class_distribution.min() / class_distribution.max()
print("Imbalance Ratio:", imbalance_ratio)
#PLOTTING THE IMBALANCE RATIO OF CLASSES
plt.figure(figsize=(8, 6))
class_distribution.plot(kind='bar', color='skyblue')
plt.title('Class Distribution')
plt.xlabel('Class')
plt.ylabel('Frequency')
plt.xticks(rotation=65)
plt.show()

#TESTING AND VALD DATASET IMPORTS
testing_path = r"E:\Genre Classification Dataset\test_data.txt"
testing_data = pd.read_csv(testing_path, sep=':::', names=['Id', 'Title', 'Description'], engine='python')
testing_data
testing_data.head(6)
testing_data.info()
testing_data.describe()
X= testing_data
X.shape
testing_data.drop('Lifestyle', axis=1, inplace=True, errors='ignore')
print(testing_data.columns)

test_soln_path = r"E:\Genre Classification Dataset\test_data_solution.txt"
test_soln_data = pd.read_csv(test_soln_path, sep=':::', names=['Id', 'Title', 'Description'], engine='python')
test_soln_data.drop(test_soln_data.columns[[0, 2]], axis=1, inplace=True)
test_soln_data.rename(columns = {'Title':'Actual Genre'}, inplace = True) 
test_soln_data.head(3)

#CHECKING NAN VALUES IN TESTING DATA
testing_data.isnull().sum()
class_distribution = testing_data['Title'].value_counts()
print("Class Distribution:")
print(class_distribution)

df_genres = training_data.drop(['Description', 'Title'], axis=1)
counts = []
categories = list(df_genres.columns.values)
for i in categories:
    counts.append((i, df_genres[i].sum()))

training_data.Genre.value_counts()


#PLOTTING THE CLASS DISTRIBUTION IN TESTING DATA
plt.figure(figsize=(16, 8))
sns.countplot(data=training_data, y='Genre', order=training_data['Genre'].value_counts().index, palette='inferno')
plt.xlabel('Count', fontsize=16, fontweight='bold')
plt.ylabel('Distribution of Genres', fontsize=14, fontweight='bold')
plt.show()
# Plot the distribution of genres using a bar plot

#PREPROCESSING DATASETS FOR TDFVECTORIZER
import seaborn as sns
import re
import nltk
import string
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem import LancasterStemmer

stemmer = LancasterStemmer()
stop_words = set(stopwords.words('english'))

def clean_text(text):
    text = text.lower()  # Lowercase all characters
    text = re.sub(r'@\S+', '', text)  # Remove Twitter handles
    text = re.sub(r'http\S+', '', text)  # Remove URLs
    text = re.sub(r'pic.\S+', '', text)
    text = re.sub(r"[^a-zA-Z+']", ' ', text)  # Keep only characters
    text = re.sub(r'\s+[a-zA-Z]\s+', ' ', text + ' ')  # Keep words with length > 1 only
    text = "".join([i for i in text if i not in string.punctuation])
    words = nltk.word_tokenize(text)
    stopwords = nltk.corpus.stopwords.words('english')  # Remove stopwords
    text = " ".join([i for i in words if i not in stopwords and len(i) > 2])
    text = re.sub("\s[\s]+", " ", text).strip()  # Remove repeated/leading/trailing spaces
    return text

training_data['Text_cleaning'] = training_data['Description'].apply(clean_text)
testing_data['Text_cleaning']  = testing_data['Description'].apply(clean_text)

training_data['length_Text_cleaning'] = training_data['Text_cleaning'].apply(len)
testing_data['length_Text_cleaning'] = testing_data['Text_cleaning'].apply(len)
print(training_data)
print(testing_data)
import warnings
warnings.filterwarnings("ignore", "use_inf_as_na")

# Visualize the distribution of text lengths
plt.figure(figsize=(8, 7))
sns.histplot(data=training_data, x='length_Text_cleaning', bins=20, kde=True, color='blue')
plt.xlabel('Length', fontsize=14, fontweight='bold')
plt.ylabel('Frequency', fontsize=14, fontweight='bold')
plt.title('Distribution of Lengths', fontsize=16, fontweight='bold')
plt.show()
plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
original_lengths = training_data['Description'].apply(len)
plt.hist(original_lengths, bins=range(0, max(original_lengths) + 100, 100), color='blue', alpha=0.7)
plt.title('Original Text Length')
plt.xlabel('Text Length')
plt.ylabel('Frequency')

plt.subplot(1, 2, 2)
cleaned_lengths = training_data['Text_cleaning'].apply(len)
plt.hist(cleaned_lengths, bins=range(0, max(cleaned_lengths) + 100, 100), color='green', alpha=0.7)
plt.title('Cleaned Text Length')
plt.xlabel('Text Length')
plt.ylabel('Frequency')
plt.tight_layout()
plt.show()

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

#testing_data.to_csv('predicted_genres.csv', index=False)
#print(testing_data)
tfidf_vectorizer = TfidfVectorizer(max_features=3000)
X_train = tfidf_vectorizer.fit_transform(training_data['Text_cleaning'])
X_test = tfidf_vectorizer.transform(testing_data['Text_cleaning'])
X = X_train
y = training_data['Genre']
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
import warnings
warnings.filterwarnings("ignore")

# Initialize and train a Multinomial Naive Bayes Classifier
clf_NB = MultinomialNB()
clf_NB.fit(X_train, y_train)
y_pred = clf_NB.predict(X_val)
accuracy = accuracy_score(y_val, y_pred)
print("Validation Accuracy:", accuracy)
print(classification_report(y_val, y_pred))

#logistic regression
clf_logreg = LogisticRegression(multi_class='multinomial', solver='sag')
clf_logreg.fit(X_train, y_train)
y_pred = clf_logreg.predict(X_val)
accuracy = accuracy_score(y_val, y_pred)
print("Validation Accuracy:", accuracy)
print(classification_report(y_val, y_pred))

# Use the trained model to make predictions on the test data
X_test_predictions = clf_logreg.predict(X_test)
testing_data['Predicted_Genre_LR'] = X_test_predictions
X_test_predictions = clf_NB.predict(X_test)
testing_data['Predicted_Genre_NB'] = X_test_predictions

testing_data.to_csv('predicted_genres.csv', index=False)
extracted_col = test_soln_data["Actual Genre"]
testing_data.insert(5, "Actual Genre", extracted_col)
# Display the 'test_data' DataFrame with predicted and actual genres
testing_data.head()
# Add actual genre column to predicted dataFrame
extracted_col = test_soln_data["Actual Genre"]
testing_data.insert(5, "real Actual Genre", extracted_col)
count_same_values_NB = (testing_data['Predicted_Genre_NB'] == testing_data['Actual Genre']).sum()
count_same_values_LR = (testing_data['Predicted_Genre_LR'] == testing_data['Actual Genre']).sum()

print("Number of samples where Naive Bayes Classifier predicted accurately:", count_same_values_NB)
print("Number of samples where Logistic Regression Classifier predicted accurately:", count_same_values_LR)

import transformers
from transformers.utils import logging
logging.set_verbosity_error()
from transformers import pipeline

from transformers import AutoTokenizer, AutoModelForSequenceClassification

# Use a pipeline as a high-level helper"
#0": "romance",
   # "1": "thriller",
    #"2": "action",
    #3": "animation",
    #"4": "comedy",
    #"5": "drama"


tokenizer = AutoTokenizer.from_pretrained("langfab/distilbert-base-uncased-finetuned-movie-genre")
model = AutoModelForSequenceClassification.from_pretrained("langfab/distilbert-base-uncased-finetuned-movie-genre")
from huggingface_hub import notebook_login
pipe = pipeline(task="text-classification", model="langfab/distilbert-base-uncased-finetuned-movie-genre")
pipe("Whether it's blocking up mouse holes, running from Landlords or making puppet shows in the bath, it's never a dull moment for The Young Professionals. Desperate to break into the online world and escape the terrors of temping, Natalie presents the lives of six housemates struggling to get on the career ladder after uni and pay their rent on time. Which is all helped along with Keara - the one with the 'real' job.")

import gradio as gr

pipe = pipeline(task="text-classification", model="langfab/distilbert-base-uncased-finetuned-movie-genre")

def classification_pipeline(text):
    # Use the text classification pipeline to predict genre
    result = pipe(text)
    scores = {genre['label']: genre['score'] for genre in result}
    return scores

# Define the Gradio interface
demo = gr.Interface(
    fn=classification_pipeline,
    inputs=gr.Textbox(label="example message"),
    outputs=gr.Label(label="Scores for each genre")
)

# Create the Gradio interface
demo.launch()
