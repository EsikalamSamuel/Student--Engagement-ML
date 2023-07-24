import pandas as pd
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier

def correlation_btw_eng_result(data):
    correlation = data['date_submitted'].corr(data['score'])
    print(f"Correlation between date_submitted and score: {correlation}") 
    print(data.describe())
    
def correlation_btw_sub_dropout(data):
    correlation = df['date_submitted'].corr(df['dropout'])
    print(f"Correlation between date_submitted and dropout: {correlation}")
    
def load_data(file_path):
    df = pd.read_csv(file_path)
    df = df.fillna(0)
    return df

def setup_dropout(data):
    data['dropout'] = data['final_result'].apply(lambda x: 1 if x == 'Withdrawn' else 0)
    del data['final_result']
    return data

def split_data(df, feature_col, target_col, test_size, random_state):
    X = df[feature_col].values.reshape(-1, 1)
    y = df[target_col]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    return X_train, X_test, y_train, y_test

def train_logistic_regression(X_train, y_train):
    logreg = LogisticRegression()
    logreg.fit(X_train, y_train)
    return logreg

def train_KNeighborsClassifier(X_train, y_train):
    kneigbour = KNeighborsClassifier()
    kneigbour.fit(X_train, y_train)
    return kneigbour

def train_random_forest(X_train, y_train):
    rf = RandomForestClassifier()
    rf.fit(X_train, y_train)
    return rf

def train_decision_tree(X_train, y_train):
    dt = DecisionTreeClassifier()
    dt.fit(X_train, y_train)
    return dt

def train_support_vector_machines(X_train, y_train):
    svm = SVR()
    svm.fit(X_train, y_train)
    return svm

def evaluate_model(y_test, y_pred, model):
    classification_rep = classification_report(y_test, y_pred)
    print(f"Classification Report for {model.upper()}:\n", classification_rep)

# # correlation between score submission
# df = load_data('~/student-engagement/Dataset/assessment_students.csv')
# #correlation_btw_eng_result(df)

# correlation between date_submitted and dropout
df = load_data('student-engagement/Dataset/studentAssessmentProcessed.csv')
setup_dropout(df)
#correlation_btw_sub_dropout(df)

# Splitting the dataset into training and testing sets
X_train, X_test, y_train, y_test = split_data(df, 'date_submitted', 'dropout', 0.2, 42)

# Training and evaluating the Logistic Regression model
logreg = train_logistic_regression(X_train, y_train)
y_pred_logreg = logreg.predict(X_test)
evaluate_model(y_test, y_pred_logreg, "Logistic Regression model BH")

# Training and evaluating the Random Forest model
rf = train_random_forest(X_train, y_train)
y_pred_rf = rf.predict(X_test)
evaluate_model(y_test, y_pred_rf, "Random Forest model BH")

# Training and evaluating the Decision Tree model
dt = train_decision_tree(X_train, y_train)
y_pred_dt = dt.predict(X_test)
evaluate_model(y_test, y_pred_dt, "Decision Tree model BH")

# Training and evaluating the K Neighbors Classifier model
kneigbour = train_KNeighborsClassifier(X_train, y_train)
y_pred_kneigbour = kneigbour.predict(X_test)
evaluate_model(y_test, y_pred_kneigbour, "K Neighbors Classifier model BH")