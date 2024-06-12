#codsoft project 2
#CREDIT CARD FRAUD DETECTION USING LOGISTIC REGRESSION, RANDOM FOREST AND XGBOOST


#LOAD THE DATASET 
import numpy as np
import pandas as pd
data_train_path = r"E:\fraudTrain.csv"
data_test_path = r"E:\fraudTest.csv"
dt_train = pd.read_csv(data_train_path)
dt_test = pd.read_csv(data_test_path)

#UNDERSTAND THE CONTENT OF THE DATASET
print(len(dt_train), len(dt_test))
com_df = pd.concat([dt_train, dt_test])
len(com_df)
com_df.info()
import seaborn as sns
import matplotlib.pyplot as plt
com_df.groupby('is_fraud').count()['amt'].plot.bar()
plt.show()

fraud_pie = dt_train["is_fraud"].value_counts()
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.pie(fraud_pie, labels=["No", "YES"], autopct="%0.0f%%", colors= 'green,red')
plt.title("pie chart for classification")
plt.tight_layout()
plt.show()
#FRAUDULENT TRANSCATIONS ARE MUCH LESS 
fraud = com_df[com_df['is_fraud'] == 1]
non_fraud = com_df[com_df['is_fraud'] == 0]
print(len(fraud), len(non_fraud))
com_df.describe()
com_df.info()
com_df.isna().sum()
#COORELATION MATRIX
sns.heatmap(com_df[[i for i in com_df.columns\
                         if com_df[i].dtype == 'int64' \
                            or com_df[i].dtype == 'float64']]\
                            .corr())
plt.show()
#coorelation significant with is_fraud and amt

#BALANCE THE DATASET
class_distribution =com_df['is_fraud'].value_counts()
print(class_distribution)
imbalance_ratio = class_distribution.min() / class_distribution.max()
print("Imbalance Ratio:", imbalance_ratio/100)
##This confirms the significant class imbalance, with the majority class being substantially larger than the minority class.
bal_df = pd.concat([fraud, non_fraud.sample(len(fraud), random_state= 42)])
bal_df.shape
bal_df.groupby('is_fraud').count()['amt'].plot.bar()
plt.show()
columns_dropped = ['Unnamed: 0',
                   'merchant', 
                   'cc_num',
                   'first', 
                   'last',
                   'gender',
                   'trans_num',
                   'unix_time',
                   'street',
                   'merch_lat',
                   'merch_long',
                   'job',
                   'zip',
                   ]

bal_df.drop(columns = columns_dropped, inplace = True)
bal_df.info()
bal_df['trans_date_trans_time'] = pd.to_datetime(bal_df['trans_date_trans_time'])
bal_df['dob'] = pd.to_datetime(bal_df['dob'])
bal_df.info()
bal_df['trans_date_trans_time'] = bal_df['trans_date_trans_time'].dt.hour
bal_df = bal_df.rename(columns = {'trans_date_trans_time': 'hour_transaction'})
def get_tod(hour):
    if 4 < hour['hour_transaction'] <= 12:
        ans = 'morning'
    elif 12 < hour['hour_transaction'] <= 20:
        ans = 'afternoon'
    elif hour['hour_transaction'] <= 4 or hour['hour_transaction'] > 20:
        ans = 'night'
    return ans
bal_df['hour_transaction'] = bal_df.apply(get_tod, axis = 1)
bal_df.head()

bal_df['dob']= bal_df['dob'].dt.year
bal_df = bal_df.rename(columns = {'dob': 'age'})
from datetime import datetime
bal_df['age'] = datetime.now().year - bal_df['age']
# Analyzing how many frauds occur for each age group
bal_df[bal_df['is_fraud'] == 1].groupby('age').count()['is_fraud']
bal_df.info()

num_features = [i for i in bal_df.columns if bal_df[i].dtype == 'int64'\
                      or bal_df[i].dtype =='int32' \
                      or bal_df[i].dtype =='float64']
cat_features = [i for i in bal_df.columns if bal_df[i].dtype == 'object']

num_features
cat_features
from sklearn.preprocessing import OrdinalEncoder
encoder = OrdinalEncoder()
encoder.fit(bal_df[cat_features])
bal_df[cat_features] = encoder.transform(bal_df[cat_features])
bal_df[['is_fraud', 'age']] = bal_df[['is_fraud', 'age']].astype('float64')
bal_df.head(5)
###scaling the dataset
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
scaled_df = scaler.fit_transform(bal_df)
scaled_df= pd.DataFrame(scaled_df)
last_column = scaled_df.shape[1]-1
print(f"Not fraud: {scaled_df[scaled_df[last_column] == 0].count()[last_column]}")
print(f"Fraud: {scaled_df[scaled_df[last_column] == 1].count()[last_column]}")
##this suggests that the dataset is perfectly balanced with equal numbers of instances for both classes after scaling.
scaled_df.head()
##we need to know the column name
scaled_df.rename(columns={last_column: 'is_fraud'}, inplace=True)
scaled_df.head()

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split

X = scaled_df.drop(columns = 'is_fraud')
y = scaled_df['is_fraud']
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = LogisticRegression()
model.fit(x_train, y_train)
def predict(model, test_set, threshold):
    predictions = model.predict(test_set)
    pred_threshold = model.predict_proba(test_set)
    test_set["prediction"] = predictions
    test_set["pred_threshold"] = (pred_threshold >= threshold)[:, 1].astype(float)
    return test_set

predict(model, x_test, 0.4)
y_test = pd.DataFrame(y_test)
x_test["real fraud"] = y_test["is_fraud"]
x_test.head(8)
print(classification_report(x_test['real fraud'], x_test['prediction']))
print(classification_report(x_test['real fraud'], x_test['pred_threshold']))

from xgboost import XGBClassifier
# XGBoost classifier model
xgb = XGBClassifier(objective='binary:logistic')
xgb.fit(x_train, y_train)
x_test = x_test.drop(columns = {'prediction','pred_threshold' ,'real fraud'})
predict(xgb, x_test, 0.4)
x_test["real fraud"] = y_test["is_fraud"]
print(classification_report(x_test['real fraud'], x_test['prediction']))
print(classification_report(x_test['real fraud'], x_test['pred_threshold']))

from sklearn.ensemble import RandomForestClassifier
rf_classifier = RandomForestClassifier(n_estimators=100, 
                                        criterion='gini', 
                                        max_depth=None, 
                                        min_samples_split=2, 
                                        min_samples_leaf=1, 
                                        max_features='sqrt', 
                                        bootstrap=True, 
                                        class_weight=None)

rf_classifier.fit(x_train, y_train)
x_test = x_test.drop(columns = {'prediction','pred_threshold' ,'real fraud'})
predict(rf_classifier, x_test, 0.4)
x_test["real fraud"] = y_test["is_fraud"]
print(classification_report(x_test['real fraud'], x_test['prediction']))
print(classification_report(x_test['real fraud'], x_test['pred_threshold']))

from sklearn.tree import DecisionTreeClassifier
dt_classifier = DecisionTreeClassifier(criterion='gini',  # Use Gini impurity as the criterion
                                       max_depth=10,       # Limit the maximum depth of the tree to prevent overfitting
                                       min_samples_split=5,  # Require at least 5 samples to split an internal node
                                       min_samples_leaf=2,   # Require at least 2 samples to be at a leaf node
                                       max_features=None,    # Consider all features when looking for the best split
                                       class_weight='balanced')
dt_classifier.fit(x_train, y_train)
x_test = x_test.drop(columns = {'prediction','pred_threshold' ,'real fraud'})
predict(dt_classifier, x_test, 0.4)
x_test["real fraud"] = y_test["is_fraud"]
print(classification_report(x_test['real fraud'], x_test['prediction']))
print(classification_report(x_test['real fraud'], x_test['pred_threshold']))







