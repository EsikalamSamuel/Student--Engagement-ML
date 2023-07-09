import pandas as pd
import csv
# Merge studentinfo and student registration
df_info = pd.read_csv('~/student-engagement/Dataset/studentInfo.csv')
df_reg = pd.read_csv('~/student-engagement/Dataset/studentRegistration.csv')
df = df_info.merge(df_reg, on=["id_student"], how='inner')

# Define common headers and fill NaN values
headers = ['id_student', 'gender', 'date_registration', 'date_unregistration', 'highest_education', 'num_of_prev_attempts', 'final_result']
df.fillna(0, inplace=True)

# Save studentInfoProcessed.csv
df.to_csv('~/student-engagement/Dataset/studentInfoProcessed.csv', index=False, columns=headers)

# Merge Student assessment with assessment and student Info
df_ass = pd.read_csv('~/student-engagement/Dataset/studentAssessment.csv')
df_assessments = pd.read_csv('~/student-engagement/Dataset/assessments.csv')
df = df.merge(df_ass, on=['id_student'], how='inner')
df = df.merge(df_assessments, on=['id_assessment'], how='inner', suffixes=('_student', '_assessment'))

# Define headers for studentAssessmentProcessed.csv
headers = ['id_student', 'final_result', 'sum_click']

# Save studentAssessmentProcessed.csv
df.to_csv('~/student-engagement/Dataset/studentAssessmentProcessed.csv', index=False, columns=['id_student', 'final_result', 'score'])

df = pd.read_csv('~/student-engagement/Dataset/studentAssessmentProcessed.csv')

# Divide studentVle.csv into four parts
df_vle = pd.read_csv('~/student-engagement/Dataset/studentVle.csv')
num_parts = 4
chunk_size = len(df_vle) // num_parts

for i in range(num_parts):
    start_index = i * chunk_size
    end_index = (i + 1) * chunk_size if i < num_parts - 1 else len(df_vle)
    df_part = df_vle[start_index:end_index]
    df_part = df_part.merge(df, on=["id_student"], how='inner')

    # Save studentVleProcessed.csv
    df_part.fillna(0, inplace=True)
    df_part.to_csv(f'~/student-engagement/Dataset/studentVleProcessed{chr(65+i)}.csv', index=False, columns=headers)
