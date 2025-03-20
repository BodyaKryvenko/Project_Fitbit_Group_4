### CLASSIFYING THE WEIGHTS ###
import pandas as pd

activity_df = pd.read_csv('daily_acivity.csv')

# Ensure Ids aren't converted to scientific notation
activity_df['Id'] = activity_df['Id'].astype(int)

user_activity_counts = activity_df['Id'].value_counts().reset_index()
user_activity_counts.columns = ['Id', 'ActivityCount']

# Function to classify users based on their activity count
def classify_user(activity_count):
    if activity_count <= 10:
        return 'Light user'
    elif 11 <= activity_count <= 15:
        return 'Moderate user'
    else:
        return 'Heavy user'

user_activity_counts['Class'] = user_activity_counts['ActivityCount'].apply(classify_user)
user_classification_df = user_activity_counts[['Id', 'Class']]
print(user_classification_df)
