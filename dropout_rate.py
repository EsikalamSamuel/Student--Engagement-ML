import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv('~/student-engagement/Dataset/studentInfo.csv')

# Calculate dropout rate
dropout_rate = len(data[data['final_result'] == 'Withdrawn']) / len(data) * 100
pass_rate = len(data[data['final_result'] == 'Pass']) / len(data) * 100
fail_rate = len(data[data['final_result'] == 'Fail']) / len(data) * 100

# Plot the dropout rate
labels = ['Dropout', 'Pass', 'Fail']
sizes = [dropout_rate, pass_rate, fail_rate]
colors = ['#ff9999', '#66b3ff', '#99ff99']

plt.bar(labels, sizes, color=colors)
plt.title('Student Outcome Distribution')
plt.xlabel('Outcome')
plt.ylabel('Percentage')
plt.ylim(0, 100)

# Add percentage values on top of each bar
for i, v in enumerate(sizes):
    plt.text(i, v + 2, f'{v:.1f}%', ha='center', color='black')

plt.savefig('check.png')