import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
import os

# Load the dataset
df = pd.read_csv('student-engagement/Dataset/studentInfoProcessed.csv')
del df['id_student']

# Perform detailed distributive analysis

# Histogram for gender
plt.figure(figsize=(6, 4))
df['gender'].hist(rwidth=0.8, color='lightgreen')
plt.xlabel('Gender')
plt.ylabel('Frequency')
plt.title('Distribution of Gender')
plt.savefig(f"Distribution of Gender.png")
plt.close()

# Histogram for date_registration
plt.figure(figsize=(6, 4))
df['date_registration'].hist(rwidth=0.8, color='lightcoral')
plt.xlabel('date_registration')
plt.ylabel('Frequency')
plt.title('Distribution of date_registration')
plt.savefig('Distribution of date_registration.png')
plt.close()

# Histogram for date_unregistration
plt.figure(figsize=(6, 4))
df['date_unregistration'].hist(rwidth=0.8, color='orange')
plt.xlabel('date_unregistration')
plt.ylabel('Frequency')
plt.title('Distribution of date_unregistration')
plt.savefig("Distribution of date_unregistration.png")
plt.close()

# Histogram for highest_education
plt.figure(figsize=(6, 4))
df['highest_education'].hist(rwidth=0.8, color='purple')
plt.xlabel('highest_education')
plt.ylabel('Frequency')
plt.title('Distribution of highest_education')
plt.savefig("Distribution of highest_education.png")
plt.close()

# Histogram for num_of_prev_attempts
plt.figure(figsize=(6, 4))
df['num_of_prev_attempts'].hist(rwidth=0.8, color='teal')
plt.xlabel('num_of_prev_attempts')
plt.ylabel('Frequency')
plt.title('Distribution of num_of_prev_attempts')
plt.savefig('Distribution of num_of_prev_attempts.png')
plt.close()

# Histogram for final_result
plt.figure(figsize=(6, 4))
df['final_result'].hist(rwidth=0.8, color='pink')
plt.xlabel('Final result')
plt.ylabel('Frequency')
plt.title('Distribution of Final result')
plt.savefig('Distribution of Final result.png')
plt.close()

# Select only the numeric columns for correlation analysis
numeric_columns = df.select_dtypes(include=['int64', 'float64']).columns

# Encode the string columns using label encoding
label_encoder = LabelEncoder()
for column in df.columns:
    if column not in numeric_columns:
        df[column] = label_encoder.fit_transform(df[column])

# Calculate standard deviation
std_values = df.std()

# Plotting the standard deviation
plt.figure(figsize=(10, 5))
sns.barplot(x=std_values.index, y=std_values.values)
plt.xticks(rotation=90)
plt.xlabel('Variables')
plt.ylabel('Standard Deviation')
plt.title('Standard Deviation of Variables')
plt.savefig('Standard Deviation of Variables.png')
plt.close()

# Calculate mean
mean_values = df.mean()

# Plotting the mean
plt.figure(figsize=(10, 5))
sns.barplot(x=mean_values.index, y=mean_values.values)
plt.xticks(rotation=90)
plt.xlabel('Variables')
plt.ylabel('Mean')
plt.title('Mean of Variables')
plt.savefig('Mean of Variables.png')
plt.close()

# Calculate mode
mode_values = df.mode().iloc[0]

# Plotting the mode
plt.figure(figsize=(10, 5))
sns.barplot(x=mode_values.index, y=mode_values.values)
plt.xticks(rotation=90)
plt.xlabel('Variables')
plt.ylabel('Mode')
plt.title('Mode of Variables')
plt.savefig('Mode of Variables.png')
plt.close()

# Calculate correlations
correlation_matrix = df.corr()

# Plotting the correlation matrix using a heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Correlation Matrix")
plt.savefig('Correlation Matrix.png')
plt.close()

num_parts = 2

for i in range(num_parts):
    df = pd.read_csv(f'student-engagement/Dataset/studentVleProcessed{chr(65+i)}.csv')
    
    del df['id_student']
    # Group the data by final_result and calculate the mean score
    mean_score_by_result = df.groupby('final_result')['score'].mean()

    # Create a bar plot to visualize the mean scores by final_result
    mean_score_by_result.plot(kind='bar', figsize=(8, 6))
    plt.xlabel('Final Result')
    plt.ylabel('Mean Score')
    plt.title('Mean Score by Final Result')
    plt.savefig(f'Mean Score by Final Result{chr(65+i)}.png')
    plt.close()