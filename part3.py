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
