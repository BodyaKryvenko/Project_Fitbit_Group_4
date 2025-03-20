### CLASSIFYING THE ACTIVITY ###
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

# ----------------------------------------------------------

### INSPECTING THE DATABASE ###
import sqlite3

conn = sqlite3.connect('fitbit_database.db')
cursor = conn.cursor()

# List tables and their columns
cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
tables = cursor.fetchall()
print("Tables in the database:", tables)

for table in tables:
    table_name = table[0]
    cursor.execute(f"PRAGMA table_info({table_name});")
    columns = cursor.fetchall()
    print(f"\nColumns in {table_name}:")
    for col in columns:
        print(col[1])

# ----------------------------------------------------------

### VERIFYING TOTAL STEPS MATCH ###
def verify_total_steps(conn):
    query = """
    SELECT 
        da.Id, 
        da.ActivityDate, 
        da.TotalSteps, 
        SUM(hs.StepTotal) AS SumHourlySteps
    FROM 
        daily_activity da
    LEFT JOIN 
        hourly_steps hs
    ON 
        da.Id = hs.Id AND da.ActivityDate = DATE(hs.ActivityHour)
    GROUP BY 
        da.Id, da.ActivityDate
    HAVING 
        da.TotalSteps != SUM(hs.StepTotal);
    """
    
    result = pd.read_sql_query(query, conn)
    
    if result.empty:
        print("Data is consistent: TotalSteps matches the sum of hourly steps for all records.")
    else:
        print("Data inconsistency found. The following records have mismatched TotalSteps:")
        print(result)

verify_total_steps(conn)
conn.close()

# ----------------------------------------------------------

### SLEEP DURATION ###
conn = sqlite3.connect('fitbit_database.db')

# Compute sleep duration for each logId
query = """
SELECT 
    Id, 
    logId, 
    COUNT(*) AS SleepDurationMinutes
FROM 
    minute_sleep
WHERE 
    value = 1
GROUP BY 
    Id, logId;
"""

sleep_duration_df = pd.read_sql_query(query, conn)

# Ensure Id and logId are integers
sleep_duration_df['Id'] = sleep_duration_df['Id'].astype(int)
sleep_duration_df['logId'] = sleep_duration_df['logId'].astype(int)

print(sleep_duration_df)
conn.close()

# ----------------------------------------------------------

import sqlite3
import pandas as pd
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import scipy.stats as stats

conn = sqlite3.connect('fitbit_database.db')

# Query to compute sleep duration for each individual and date
sleep_query = """
SELECT 
    Id,
    date,
    value,
    logId
FROM 
    minute_sleep;
"""

sleep_df = pd.read_sql_query(sleep_query, conn)
sleep_df['Id'] = sleep_df['Id'].astype(int)
sleep_df['date'] = pd.to_datetime(sleep_df['date']).dt.date

# Filter rows where value = 1 (indicating sleep)
sleep_df = sleep_df[sleep_df['value'] == 1]

sleep_duration_df = sleep_df.groupby(['Id', 'date']).size().reset_index(name='SleepDurationMinutes')

# ----------------------------------------------------------

### DURATION OF SLEEP TO ACTIVE MINUTES ###
# Compute total active minutes for each individual and date
active_minutes_query = """
SELECT 
    Id,
    ActivityDate,
    (VeryActiveMinutes + FairlyActiveMinutes + LightlyActiveMinutes) AS TotalActiveMinutes
FROM 
    daily_activity;
"""

active_minutes_df = pd.read_sql_query(active_minutes_query, conn)
active_minutes_df['Id'] = active_minutes_df['Id'].astype(int)
active_minutes_df['ActivityDate'] = pd.to_datetime(active_minutes_df['ActivityDate']).dt.date

merged_df = pd.merge(
    active_minutes_df,
    sleep_duration_df,
    left_on=['Id', 'ActivityDate'],
    right_on=['Id', 'date'],
    how='inner'
)

merged_df = merged_df[['Id', 'ActivityDate', 'TotalActiveMinutes', 'SleepDurationMinutes']]
merged_df = merged_df.rename(columns={'ActivityDate': 'Date'})

print("Activity and sleep dataframe")
print(merged_df.head())

# Perform linear regression
X = merged_df[['TotalActiveMinutes']]
y = merged_df['SleepDurationMinutes']

model = LinearRegression()
model.fit(X, y)

print(f"Intercept: {model.intercept_}")
print(f"Coefficient: {model.coef_[0]}")
print(f"R-squared: {model.score(X, y)}")

# Calculate residuals
residuals = y - model.predict(X)

plt.figure(figsize=(12, 5))

# Scatter plot
plt.subplot(1, 2, 1)
plt.scatter(X, y, color='blue')
plt.plot(X, model.predict(X), color='red')
plt.xlabel('Total Active Minutes')
plt.ylabel('Sleep Duration Minutes')
plt.title('Regression of Sleep Duration on Total Active Minutes')
plt.legend()

# Q-Q plot for residuals
plt.subplot(1, 2, 2)
stats.probplot(residuals, dist="norm", plot=plt)
plt.title('Q-Q Plot of Residuals')
plt.xlabel('Theoretical Quantiles')
plt.ylabel('Sample Quantiles')

plt.tight_layout()
plt.show(block=False)

# ----------------------------------------------------------

### DURATION OF SLEEP TO SEDENTARY MINUTES ###
sedentary_query = """
SELECT 
    Id,
    ActivityDate,
    SedentaryMinutes
FROM 
    daily_activity;
"""

sedentary_df = pd.read_sql_query(sedentary_query, conn)
sedentary_df['Id'] = sedentary_df['Id'].astype(int)
sedentary_df['ActivityDate'] = pd.to_datetime(sedentary_df['ActivityDate']).dt.date

merged_df = pd.merge(
    sedentary_df,
    sleep_duration_df,
    left_on=['Id', 'ActivityDate'],
    right_on=['Id', 'date'],
    how='inner'
)

merged_df = merged_df[['Id', 'ActivityDate', 'SedentaryMinutes', 'SleepDurationMinutes']]
merged_df = merged_df.rename(columns={'ActivityDate': 'Date'})

# Perform linear regression
X = merged_df[['SedentaryMinutes']]
y = merged_df['SleepDurationMinutes']

model = LinearRegression()
model.fit(X, y)

print(f"\nIntercept: {model.intercept_}")
print(f"Coefficient: {model.coef_[0]}")
print(f"R-squared: {model.score(X, y)}")

residuals = y - model.predict(X)

plt.figure(figsize=(12, 5))

# Scatter plot
plt.subplot(1, 2, 1)
plt.scatter(X, y, color='blue')
plt.plot(X, model.predict(X), color='red')
plt.xlabel('Sedentary Minutes')
plt.ylabel('Sleep Duration Minutes')
plt.title('Regression of Sleep Duration on Sedentary Minutes')
plt.legend()

# Q-Q plot
plt.subplot(1, 2, 2)
stats.probplot(residuals, dist="norm", plot=plt)
plt.title('Q-Q Plot of Residuals')
plt.xlabel('Theoretical Quantiles')
plt.ylabel('Sample Quantiles')

plt.tight_layout()
plt.show(block=False)

conn.close()