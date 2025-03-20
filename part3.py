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

# -----------------------------------------
conn = sqlite3.connect('fitbit_database.db')

# Define the 4-hour blocks
time_blocks = {
    '0-4': (0, 4),
    '4-8': (4, 8),
    '8-12': (8, 12),
    '12-16': (12, 16),
    '16-20': (16, 20),
    '20-24': (20, 24)
}

chronological_order = ['0-4', '4-8', '8-12', '12-16', '16-20', '20-24']

# Function to assign a time block based on the hour
def assign_time_block(hour):
    for block, (start, end) in time_blocks.items():
        if start <= hour < end:
            return block
    return None

# ----------------------------------------------------------
### AVERAGE STEPS PER TIME BLOCK ###

hourly_steps_query = """
SELECT 
    Id, 
    ActivityHour, 
    StepTotal
FROM 
    hourly_steps;
"""

hourly_steps_df = pd.read_sql_query(hourly_steps_query, conn)

hourly_steps_df['ActivityHour'] = pd.to_datetime(hourly_steps_df['ActivityHour'])
hourly_steps_df['Hour'] = hourly_steps_df['ActivityHour'].dt.hour

hourly_steps_df['TimeBlock'] = hourly_steps_df['Hour'].apply(assign_time_block)

# Average steps per time block
average_steps_per_block = hourly_steps_df.groupby('TimeBlock')['StepTotal'].mean().reset_index()

average_steps_per_block['TimeBlock'] = pd.Categorical(average_steps_per_block['TimeBlock'], categories=chronological_order, ordered=True)
average_steps_per_block = average_steps_per_block.sort_values('TimeBlock')

plt.figure(figsize=(10, 6))
plt.bar(average_steps_per_block['TimeBlock'], average_steps_per_block['StepTotal'], color='blue')
plt.xlabel('Time Block')
plt.ylabel('Average Steps')
plt.title('Average Steps per 4-Hour Time Block')
plt.show(block=False)

# ----------------------------------------------------------
### AVERAGE CALORIES BURNT PER TIME BLOCK ###

hourly_calories_query = """
SELECT 
    Id, 
    ActivityHour, 
    Calories
FROM 
    hourly_calories;
"""

hourly_calories_df = pd.read_sql_query(hourly_calories_query, conn)

hourly_calories_df['ActivityHour'] = pd.to_datetime(hourly_calories_df['ActivityHour'])
hourly_calories_df['Hour'] = hourly_calories_df['ActivityHour'].dt.hour

hourly_calories_df['TimeBlock'] = hourly_calories_df['Hour'].apply(assign_time_block)

# Average calories burnt per time block
average_calories_per_block = hourly_calories_df.groupby('TimeBlock')['Calories'].mean().reset_index()

average_calories_per_block['TimeBlock'] = pd.Categorical(average_calories_per_block['TimeBlock'], categories=chronological_order, ordered=True)
average_calories_per_block = average_calories_per_block.sort_values('TimeBlock')

plt.figure(figsize=(10, 6))
plt.bar(average_calories_per_block['TimeBlock'], average_calories_per_block['Calories'], color='green')
plt.xlabel('Time Block')
plt.ylabel('Average Calories Burnt')
plt.title('Average Calories Burnt per 4-Hour Time Block')
plt.show(block=False)

# ----------------------------------------------------------
### AVERAGE SLEEP MINUTES PER TIME BLOCK ###

minute_sleep_query = """
SELECT 
    Id, 
    date, 
    value
FROM 
    minute_sleep;
"""

minute_sleep_df = pd.read_sql_query(minute_sleep_query, conn)

minute_sleep_df['date'] = pd.to_datetime(minute_sleep_df['date'])
minute_sleep_df['Hour'] = minute_sleep_df['date'].dt.hour

minute_sleep_df['TimeBlock'] = minute_sleep_df['Hour'].apply(assign_time_block)

minute_sleep_df = minute_sleep_df[minute_sleep_df['value'] == 1]

# Average sleep minutes per time block
average_sleep_per_block = minute_sleep_df.groupby('TimeBlock').size().reset_index(name='SleepMinutes')
average_sleep_per_block['SleepMinutes'] /= len(minute_sleep_df['Id'].unique())

average_sleep_per_block['TimeBlock'] = pd.Categorical(average_sleep_per_block['TimeBlock'], categories=chronological_order, ordered=True)
average_sleep_per_block = average_sleep_per_block.sort_values('TimeBlock')

plt.figure(figsize=(10, 6))
plt.bar(average_sleep_per_block['TimeBlock'], average_sleep_per_block['SleepMinutes'], color='purple')
plt.xlabel('Time Block')
plt.ylabel('Average Sleep Minutes')
plt.title('Average Sleep Minutes per 4-Hour Time Block')
plt.show(block=False)

conn.close()

# ----------------------------------------------------------

### PLOT HEART RATE AND INTENSITY ###
def plot_heart_rate_and_intensity(individual_id):
    """
    Plots the heart rate and total intensity of exercise for a given individual.
    """
    conn = sqlite3.connect('fitbit_database.db')

    heart_rate_query = f"""
    SELECT 
        Time, 
        Value AS HeartRate
    FROM 
        heart_rate
    WHERE 
        Id = {individual_id};
    """

    heart_rate_df = pd.read_sql_query(heart_rate_query, conn)

    if not heart_rate_df.empty:
        heart_rate_df['Time'] = pd.to_datetime(heart_rate_df['Time'])

    # Query intensity data for the given individual
    intensity_query = f"""
    SELECT 
        ActivityHour, 
        TotalIntensity
    FROM 
        hourly_intensity
    WHERE 
        Id = {individual_id};
    """

    intensity_df = pd.read_sql_query(intensity_query, conn)

    if not intensity_df.empty:
        intensity_df['ActivityHour'] = pd.to_datetime(intensity_df['ActivityHour'])

    conn.close()

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

    # Plot heart rate
    if not heart_rate_df.empty:
        ax1.plot(heart_rate_df['Time'], heart_rate_df['HeartRate'], color='red', label='Heart Rate')
        ax1.set_ylabel('Heart Rate (bpm)')
        ax1.set_title(f'Heart Rate and Exercise Intensity for Individual {individual_id}')
        ax1.legend()
        ax1.grid(True)

    # Plot total intensity
    if not intensity_df.empty:
        ax2.plot(intensity_df['ActivityHour'], intensity_df['TotalIntensity'], color='blue', label='Total Intensity')
        ax2.set_xlabel('Time')
        ax2.set_ylabel('Total Intensity')
        ax2.legend()
        ax2.grid(True)

    plt.tight_layout()
    return fig

# Find all Ids with heart rate data
conn = sqlite3.connect('fitbit_database.db')
heart_rate_ids_query = """
SELECT DISTINCT CAST(Id AS INTEGER) AS Id
FROM heart_rate;
"""
heart_rate_ids_df = pd.read_sql_query(heart_rate_ids_query, conn)
conn.close()

print("Ids with heart rate data:")
print(heart_rate_ids_df)

# Pick an Id with heart rate data
if not heart_rate_ids_df.empty:
    individual_id = heart_rate_ids_df['Id'].iloc[0]
    print(f"Selected Id: {individual_id}")

    fig = plot_heart_rate_and_intensity(individual_id)
    plt.show()
else:
    print("No heart rate data found in the database.")


#------------------------------------------------------------------------
weather_df = pd.read_csv('chicago.csv')

weather_df = weather_df.iloc[:,[1, 2, 3, 4, 5, 6, 7]]

conn = sqlite3.connect('fitbit_database.db')

intensity_query = f"""
    SELECT 
        ActivityHour, 
        TotalIntensity AS TotalIntensity
    FROM 
        hourly_intensity
    """

intensity_df = pd.read_sql_query(intensity_query, conn)

intensity_df = intensity_df.groupby(by="ActivityHour").sum()
intensity_df = intensity_df.reset_index()

intensity_df['datetime'] = pd.to_datetime(intensity_df['ActivityHour']).dt.strftime('%Y-%m-%dT%H:%M:%S')

def variable_comparison(x):
    value_df = weather_df.iloc[:, [0, x]]  

    df = pd.merge(value_df, intensity_df, how='left', on='datetime')
    df = df.dropna()
    
    X = df[['TotalIntensity']]
    y = df.iloc[:, 1]

    model = LinearRegression()
    model.fit(X, y)
    
    print(f"\nIntercept: {model.intercept_}")
    print(f"Coefficient: {model.coef_[0]}")
    print(f"R-squared: {model.score(X, y)}")

    residuals = y - model.predict(X)
    
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.scatter(X, y, color='blue', label='Data points')
    plt.plot(X, model.predict(X), color='red', label='Regression line')
    plt.xlabel('TotalIntensity')
    plt.ylabel({weather_df.columns[x]})
    plt.title('Regression')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    stats.probplot(residuals, dist="norm", plot=plt)
    plt.title('Q-Q Plot of Residuals')
    plt.xlabel('Theoretical Quantiles')
    plt.ylabel('Sample Quantiles')

    plt.tight_layout()
    plt.show(block=False)
    
    return df

conn.close()

x = variable_comparison(1) #Use 1-6 for different variables