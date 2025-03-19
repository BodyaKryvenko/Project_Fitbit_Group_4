import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
import numpy as np

file_path = 'daily_acivity.csv'
df = pd.read_csv(file_path)
print(df.head())

unique_users = df['Id'].nunique()
print(f'There are {unique_users} unique users in the dataset.')
total_distance_per_user = df.groupby('Id')['TotalDistance'].sum()

# Plot total distance per user
total_distance_per_user.plot(kind='bar', figsize=(10, 6))
plt.title('Total Distance per User')
plt.xlabel('User ID')
plt.ylabel('Total Distance')