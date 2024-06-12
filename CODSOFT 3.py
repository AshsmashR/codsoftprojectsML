import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt  
import seaborn as sns 

import warnings
warnings.filterwarnings('ignore')

import imblearn
from collections import Counter
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline

from sklearn.feature_extraction.text import TfidfVectorizer

data = pd.read_csv(r"E:\spam.csv", encoding='latin')
data.head()
data.info()
data.describe()
data.drop(columns=['Unnamed: 2', 'Unnamed: 3', 'Unnamed: 4'], inplace=True)
data.info()

data.rename(columns = {'v1': 'type of message', 'v2': 'Messages'}, inplace = True)

from sklearn.preprocessing import LabelEncoder
encoder = LabelEncoder()
data['type of message'] = encoder.fit_transform(data['type of message'])
data.head()
data.shape
data.isnull().sum()

firstvisualization = data['type of message'].value_counts()
colors = ['red','blue']

fig = plt.subplots(nrows = 1,ncols = 2,figsize = (20,5))
plt.subplot(1,2,1)
firstvisualization.plot.pie(explode=[0,0.1],autopct='%1.1f%%',shadow=True, labels=['geniune', 'Spam'], colors=colors)
plt.title('target (%)')
plt.show()
plt.subplot(1,2,2)
ax = sns.countplot(x='type of message',data = data, palette = colors,edgecolor = 'black', width=0.4)
for rect in ax.patches:
    ax.text(rect.get_x() + rect.get_width() / 2, rect.get_height(), rect.get_height(), horizontalalignment='center', fontsize = 11)
ax.set_xticklabels(['Ham', 'Spam'])
plt.title('Number of Target')
plt.show()

x = data['Messages']
y = data['type of message']
feature_extraction = TfidfVectorizer(min_df = 1, stop_words='english', lowercase=True)

x_extraction = feature_extraction.fit_transform(x)
print(x_extraction)

over = SMOTE(sampling_strategy = 1)
under = RandomUnderSampler(sampling_strategy = 0.4)
f1 = x_extraction
t1 = y

steps = [('under', under),('over', over)]
pipeline = Pipeline(steps=steps)
f1, t1 = pipeline.fit_resample(f1, t1)
Counter(t1)

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.metrics import precision_recall_curve


from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
# Splitting the resampled data into training and testing sets
x=data['Messages']
y=data['type of message']
x_train, x_test, y_train, y_test = train_test_split(f1, t1, test_size=0.3, random_state=101)
# Defining a function to train and evaluate a machine learning classifier model

def model(classifier,x_train,y_train,x_test,y_test):

    classifier.fit(x_train,y_train)
    prediction = classifier.predict(x_test)
    cv = RepeatedStratifiedKFold(n_splits = 10,n_repeats = 3,random_state = 1)#kflod for better accuracy of mlmodel
    print("Cross Validation Score : ",'{0:.2%}'.format(cross_val_score(classifier,x_train,y_train,cv = cv,scoring = 'roc_auc').mean()))
    print("ROC_AUC Score : ",'{0:.2%}'.format(roc_auc_score(y_test,prediction)))



# Defining a function to evaluate the performance of a machine learning classifier model
def model_evaluation(classifier,x_test,y_test):

    # Confusion Matrix
    cm = confusion_matrix(y_test,classifier.predict(x_test))
    names = ['True Neg','False Pos','False Neg','True Pos']
    counts = [value for value in cm.flatten()]
    percentages = ['{0:.2%}'.format(value) for value in cm.flatten()/np.sum(cm)]
    labels = [f'{v1}\n{v2}\n{v3}' for v1, v2, v3 in zip(names,counts,percentages)]
    labels = np.asarray(labels).reshape(2,2)
    sns.heatmap(cm,annot = labels,cmap = colors,fmt ='')

    # Classification Report
    print(classification_report(y_test,classifier.predict(x_test)))





# Defining a function to plot the Receiver Operating Characteristic (ROC) curve
def plot_roc_curve(y_true, y_scores):
    # Calculate the false positive rate (FPR) and true positive rate (TPR)
    fpr, tpr, _ = roc_curve(y_true, y_scores)

    # Calculate the area under the ROC curve (AUC)
    auc = roc_auc_score(y_true, y_scores)

    # Plot the ROC curve
    plt.figure()
    plt.plot(fpr, tpr, label='ROC curve (AUC = {:.2f})'.format(auc))
    plt.plot([0, 1], [0, 1], 'k--')  # Diagonal line (random classifier)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend(loc='lower right')
    plt.show()


classifier_xgb = XGBClassifier(learning_rate= 0.01 ,max_depth = 2,n_estimators = 1000)
# Training, evaluating, and evaluating the performance of an XGBoost classifier model

model(classifier_xgb,x_train,y_train,x_test,y_test)
model_evaluation(classifier_xgb,x_test,y_test)
prediction_xgb = classifier_xgb.predict(x_test)
plot_roc_curve(y_test, prediction_xgb)


classifier_lr = LogisticRegression(random_state = 0,C=10,penalty= 'l2')
# Training, evaluating, and evaluating the performance of a Logistic Regression classifier model

model(classifier_lr,x_train,y_train,x_test,y_test)
model_evaluation(classifier_lr,x_test,y_test) 
prediction_lr = classifier_lr.predict(x_test)
plot_roc_curve(y_test, prediction_lr)
from sklearn.naive_bayes import BernoulliNB, MultinomialNB, ComplementNB

classifer = BernoulliNB()
classifer.fit(x_train, y_train)
model(classifer,x_train,y_train,x_test,y_test)
model_evaluation(classifer,x_test,y_test) 
prediction_lr = classifer.predict(x_test)
plot_roc_curve(y_test, prediction_lr)


import transformers
from transformers.utils import logging
logging.set_verbosity_error()
from transformers import pipeline
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import warnings

def fxn():
    warnings.warn("deprecated", DeprecationWarning)

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    fxn()
tokenizer = AutoTokenizer.from_pretrained("dima806/sms-spam-detection-distilbert")
model = AutoModelForSequenceClassification.from_pretrained("dima806/sms-spam-detection-distilbert")
from huggingface_hub import notebook_login
pipe = pipeline(task="text-classification", model="dima806/sms-spam-detection-distilbert")
pipe("WINNER!! As a valued network customer you have been selected to receivea å£900 prize reward! To claim call 09061701461. Claim code KL341. Valid 12 hours only")

import os 
import gradio as gr
from transformers import pipeline

# Initialize the sentiment analysis pipeline
sms_classify_pipe = pipeline(task="text-classification", model="dima806/sms-spam-detection-distilbert")

def sms_classify(text):
    # Use the sentiment analysis pipeline to predict sentiment
    result = sms_classify_pipe(text)
    spam_confidence = result[0]['score']
    return {'Spam': spam_confidence}

# Define the Gradio interface
demo = gr.Interface(
    fn=sms_classify,
    inputs=gr.Textbox(label="example message"),
    outputs=gr.Label(label="Spam Confidence Score")
)

# Create the Gradio interface
demo.launch() #share=True, server_port=int(os.environ.get('PORT1', 9090)))
