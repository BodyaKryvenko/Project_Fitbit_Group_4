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

# ----------------------------------------------------------------------

def plot_calories_burnt(user_id, start_date=None, end_date=None):
    df['ActivityDate'] = pd.to_datetime(df['ActivityDate'])
    
    user_data = df[df['Id'] == user_id]
    
    if start_date:
        start_date = datetime.strptime(start_date, '%Y-%m-%d')
        user_data = user_data[user_data['ActivityDate'] >= start_date]
    if end_date:
        end_date = datetime.strptime(end_date, '%Y-%m-%d')
        user_data = user_data[user_data['ActivityDate'] <= end_date]
    
    # Plot calories burnt per day
    plt.figure(figsize=(10, 6))
    plt.plot(user_data['ActivityDate'], user_data['Calories'], marker='o')
    plt.title(f'Calories Burnt per Day for User {user_id}')
    plt.xlabel('Date')
    plt.ylabel('Calories Burnt')
    plt.xticks(rotation=45)
    plt.grid(True)
    plt.show()

# Example usage
plot_calories_burnt(user_id=1503960366, start_date='2016-03-31', end_date='2016-04-07')
