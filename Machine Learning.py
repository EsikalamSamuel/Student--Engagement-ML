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
    correlation = df['sum_click'].corr(df['dropout'])
    print(f"Correlation between sum_click and dropout: {correlation}") 
    
    
def load_data(file_path):
    df = pd.read_csv(file_path)[0:100000]
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

def plot_actual_vs_predicted(X_test, y_test, y_pred, model):
    pass_count = len(y_pred[y_pred == 0])
    withdrawn_count = len(y_pred[y_pred == 1])
    total_count = len(y_pred)

    # Create a pie chart
    plt.figure(figsize=(8, 8))
    plt.pie([pass_count, withdrawn_count], labels=['Pass', 'Withdrawn'], autopct='%1.1f%%', colors=['blue', 'red'])
    plt.title(f'Proportion of Predicted Outcomes for {model}')

    plt.savefig(f'student-engagement/plot/{model}_pie_chart.png')
    plt.close()


def plot_countplot(X_test, y_test, y_pred, model):
    df_results = pd.DataFrame({'sum_click': X_test.flatten(), 'y_test': y_test, 'y_pred': y_pred})
    df_results['final_result_actual'] = df_results['y_test'].apply(lambda x: 'Withdrawn' if x == 1 else 'Pass')
    df_results['final_result_predicted'] = df_results['y_pred'].apply(lambda x: 'Withdrawn' if x == 1 else 'Pass')

    grouped_data = df_results.groupby(['final_result_actual', 'final_result_predicted']).size().unstack()

    plt.figure(figsize=(8, 6))
    grouped_data.plot(kind='bar', stacked=True, color=['blue', 'red'], alpha=0.7)
    plt.xlabel('Final Result (Actual)')
    plt.ylabel('Count')
    plt.title(f'Stacked Bar Chart: Actual vs. Predicted Outcomes for {model}')
    plt.legend(title='Predicted', loc='upper right', labels=['Pass', 'Withdrawn'])
    plt.savefig(f'student-engagement/plot/{model}_stacked_bar_chart.png')
    plt.close()

def plot_confusion_matrix(y_test, y_pred, class_names, model):
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
                xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.savefig(f'student-engagement/plot/{model}_confusion_matrix.png')
    plt.close()

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

# Load the dataset
df = load_data('~/student-engagement/Dataset/studentVleProcessedC.csv')

# Setting up dropout column
df = setup_dropout(df)

correlation_btw_eng_result(df)

# Splitting the dataset into training and testing sets
X_train, X_test, y_train, y_test = split_data(df, 'sum_click', 'dropout', 0.2, 42)

# Training and evaluating the Logistic Regression model
logreg = train_logistic_regression(X_train, y_train)
y_pred_logreg = logreg.predict(X_test)
plot_actual_vs_predicted(X_test, y_test, y_pred_logreg, "Logistic Regression model")
plot_countplot(X_test, y_test, y_pred_logreg, "Logistic Regression model")
plot_confusion_matrix(y_test, y_pred_logreg, ['Pass', 'Withdrawn'], "Logistic Regression model")
evaluate_model(y_test, y_pred_logreg, "Logistic Regression model")

# Training and evaluating the Random Forest model
rf = train_random_forest(X_train, y_train)
y_pred_rf = rf.predict(X_test)
plot_actual_vs_predicted(X_test, y_test, y_pred_rf, "Random Forest model")
plot_countplot(X_test, y_test, y_pred_rf, "Random Forest model")
plot_confusion_matrix(y_test, y_pred_rf, ['Pass', 'Withdrawn'], "Random Forest model")
evaluate_model(y_test, y_pred_rf, "Random Forest model")

# Training and evaluating the Decision Tree model
dt = train_decision_tree(X_train, y_train)
y_pred_dt = dt.predict(X_test)
plot_actual_vs_predicted(X_test, y_test, y_pred_dt, "Decision Tree model")
plot_countplot(X_test, y_test, y_pred_dt, "Decision Tree model")
plot_confusion_matrix(y_test, y_pred_dt, ['Pass', 'Withdrawn'], "Decision Tree model")
evaluate_model(y_test, y_pred_dt, "Decision Tree model")

# Training and evaluating the K Neighbors Classifier model
kneigbour = train_KNeighborsClassifier(X_train, y_train)
y_pred_kneigbour = kneigbour.predict(X_test)
plot_actual_vs_predicted(X_test, y_test, y_pred_kneigbour, "K Neighbors Classifier model")
plot_countplot(X_test, y_test, y_pred_kneigbour, "K Neighbors Classifier model")
plot_confusion_matrix(y_test, y_pred_kneigbour, ['Pass', 'Withdrawn'], "K Neighbors Classifier model")
evaluate_model(y_test, y_pred_kneigbour, "K Neighbors Classifier model")
