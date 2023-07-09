import pandas as pd

# Load the dataset
df_student_assessment = pd.read_csv('Dataset/studentAssessmentProcessed.csv')
df_student_vle = pd.read_csv('Dataset/studentVleProcessedA.csv')

# Get summary statistics of numerical variables
print(df_student_assessment.describe())
print(df_student_vle.describe())