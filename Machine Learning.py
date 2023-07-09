import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeClassifier

def load_data(file_path, num_rows):
    df = pd.read_csv(file_path)[:num_rows]
    return df

def encode_categorical(df, target_col):
    encoder = LabelEncoder()
    df[target_col] = encoder.fit_transform(df[target_col])
    return df, encoder

def split_data(df, feature_col, target_col, test_size, random_state):
    X = df[feature_col].values.reshape(-1, 1)
    y = df[target_col]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    return X_train, X_test, y_train, y_test

def plot_actual_vs_predicted(X_test, y_test_trans, y_pred_trans, model):
    sorted_indices = X_test.flatten().argsort()
    X_test_sorted = X_test[sorted_indices]
    y_test_sorted = y_test_trans[sorted_indices]
    y_pred_sorted = y_pred_trans[sorted_indices]

    plt.figure(figsize=(10, 6))
    plt.plot(X_test_sorted, y_test_sorted, color='blue', label='Actual')
    plt.plot(X_test_sorted, y_pred_sorted, color='red', label='Predicted')
    plt.xlabel('sum_click')
    plt.ylabel('final_result')
    plt.title('Actual vs. Predicted')
    plt.legend()
    plt.savefig(f'{model}_line_plot.png')
    plt.close()

def plot_confusion_matrix(y_test_trans, y_pred_trans, class_names, model):
    cm = confusion_matrix(y_test_trans, y_pred_trans)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
                xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.savefig(f'{model}_confusion_matrix.png')
    plt.close()

def train_logistic_regression(X_train, y_train):
    logreg = LogisticRegression()
    logreg.fit(X_train, y_train)
    return logreg

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

def evaluate_model(y_test_trans, y_pred_trans):
    classification_rep = classification_report(y_test_trans, y_pred_trans)
    print("Classification Report:\n", classification_rep)

# Load the dataset
df = load_data('~/student-engagement/Dataset/studentVleProcessedA.csv', 50000)

# Encoding categorical variables
df, encoder = encode_categorical(df, 'final_result')

# Splitting the dataset into training and testing sets
X_train, X_test, y_train, y_test = split_data(df, 'sum_click', 'final_result', 0.2, 42)

# Training and evaluating the Logistic Regression model
logreg = train_logistic_regression(X_train, y_train)
y_pred_logreg = logreg.predict(X_test)
y_pred_trans_logreg = encoder.inverse_transform(y_pred_logreg)
y_test_trans_logreg = encoder.inverse_transform(y_test)
plot_actual_vs_predicted(X_test, y_test_trans_logreg, y_pred_trans_logreg, "Logistic Regression model")
plot_confusion_matrix(y_test_trans_logreg, y_pred_trans_logreg, encoder.classes_, "Logistic Regression model")
evaluate_model(y_test_trans_logreg, y_pred_trans_logreg)

# Training and evaluating the Random Forest model
rf = train_random_forest(X_train, y_train)
y_pred_rf = rf.predict(X_test)
y_pred_trans_rf = encoder.inverse_transform(y_pred_rf)
y_test_trans_rf = encoder.inverse_transform(y_test)
plot_actual_vs_predicted(X_test, y_test_trans_rf, y_pred_trans_rf, "Random Forest model")
plot_confusion_matrix(y_test_trans_rf, y_pred_trans_rf, encoder.classes_, "Random Forest model")
evaluate_model(y_test_trans_rf, y_pred_trans_rf)

# Training and evaluating the Decision Tree model
dt = train_decision_tree(X_train, y_train)
y_pred_dt = dt.predict(X_test)
y_pred_trans_dt = encoder.inverse_transform(y_pred_dt)
y_test_trans_dt = encoder.inverse_transform(y_test)
plot_actual_vs_predicted(X_test, y_test_trans_dt, y_pred_trans_dt, "Decision Tree model")
plot_confusion_matrix(y_test_trans_dt, y_pred_trans_dt, encoder.classes_, "Decision Tree model")
evaluate_model(y_test_trans_dt, y_pred_trans_dt)

# Training and evaluating the Support Vector Machines model
svm = train_support_vector_machines(X_train, y_train)
y_pred_svm = svm.predict(X_test)
y_pred_trans_svm = encoder.inverse_transform(y_pred_svm)
y_test_trans_svm = encoder.inverse_transform(y_test)
plot_actual_vs_predicted(X_test, y_test_trans_svm, y_pred_trans_svm, "Support Vector Machines model")
plot_confusion_matrix(y_test_trans_svm, y_pred_trans_svm, encoder.classes_, "Support Vector Machines model")
evaluate_model(y_test_trans_svm, y_pred_trans_svm)