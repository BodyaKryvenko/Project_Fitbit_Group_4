import matplotlib
matplotlib.use('TkAgg')
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
import numpy as np
import statsmodels.formula.api as smf
import statsmodels.api as sm

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

# ----------------------------------------------------------------------

df['DayOfWeek'] = df['ActivityDate'].dt.day_name()
workout_frequency = df['DayOfWeek'].value_counts().sort_index()

# Plot the frequency of workouts for each day of the week
plt.figure(figsize=(10, 6))
workout_frequency.plot(kind='bar')
plt.title('Frequency of Workouts by Day of the Week')
plt.xlabel('Day of the Week')
plt.ylabel('Frequency of Workouts')
plt.xticks(rotation=45)
plt.grid(True)
plt.show()


# ----------------------------------------------------------------------
def user_regression(user_id):
    model = smf.ols("Calories ~ TotalSteps + C(Id)", data=df).fit()
    print(model.summary())

    user_data = df[df['Id'] == user_id]

    plt.figure(figsize=(10, 6))

    plt.scatter(user_data['TotalSteps'], user_data['Calories'], alpha=0.5, label='Data points')

    X = user_data['TotalSteps']
    X = sm.add_constant(X)
    y = user_data['Calories']

    model_user = sm.OLS(y, X).fit()

    x_range = np.linspace(X['TotalSteps'].min(), X['TotalSteps'].max(), 100)
    X_pred = sm.add_constant(x_range)
    y_pred = model_user.predict(X_pred)

    plt.plot(x_range, y_pred, 'r-', linewidth=2)
    plt.title(f'Relationship between Steps and Calories for User {user_id}')
    plt.xlabel('Total Steps')
    plt.ylabel('Calories Burnt')
    plt.grid(True, alpha=0.3)

    equation = f"Calories = {model_user.params[0]:.2f} + {model_user.params[1]:.4f} * Steps"
    r_squared = f"R² = {model_user.rsquared:.3f}"
    plt.annotate(
                equation + "\n" + r_squared,
                xy=(0.05, 0.95),
                xycoords='axes fraction',
                fontsize=10,
                bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8
                ))

    plt.tight_layout()
    plt.show()

user_regression(1503960366)
