import sqlite3
import matplotlib
matplotlib.use('TkAgg')

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import scipy.stats as stats

# Connect to the SQLite database
conn = sqlite3.connect('fitbit_database.db')
cursor = conn.cursor()


# Query the weight_log table
cursor.execute("SELECT * FROM weight_log;")
rows = cursor.fetchall()

# Get column names
cursor.execute("PRAGMA table_info(weight_log);")
columns = cursor.fetchall()
column_names = [column[1] for column in columns]

# Print the column names
print("Columns in weight_log table:", column_names)

# Print the rows with missing values highlighted
print("\nData in weight_log table (missing values highlighted):")
for row in rows:
    formatted_row = []
    for i, value in enumerate(row):
        if value is None:  # Check for NULL values
            formatted_row.append(f"[MISSING: {column_names[i]}]")
        else:
            formatted_row.append(str(value))
    print(tuple(formatted_row))

# Close the connection
conn.close()

#---------------------

# Connect to the SQLite database
conn = sqlite3.connect('fitbit_database.db')
cursor = conn.cursor()

# Query the weight_log table
cursor.execute("SELECT * FROM weight_log;")
rows = cursor.fetchall()

# Change missing WeightKg using the average of non-missing values
cursor.execute("""
    UPDATE weight_log
    SET WeightKg = (SELECT AVG(WeightKg) FROM weight_log WHERE WeightKg IS NOT NULL)
    WHERE WeightKg IS NULL;
""")

cursor.execute("""
    UPDATE weight_log
    SET Fat = (1.2 * BMI + 6.9 - 10.8)
    WHERE Fat IS NULL;
""")

# Commit the changes to the database
conn.commit()

# Verify the changes
cursor.execute("SELECT * FROM weight_log;")
updated_rows = cursor.fetchall()
print("\nUpdated weight_log table:")
for row in updated_rows:
    print(row)

# Close the connection
conn.close()


#---------------------


conn = sqlite3.connect('fitbit_database.db')
cursor = conn.cursor()

# Query the weight_log table
cursor.execute("SELECT * FROM weight_log;")
rows = cursor.fetchall()

query = """
SELECT 
    hc.Id, 
    hc.ActivityHour, 
    hc.Calories, 
    hi.TotalIntensity, 
    hi.AverageIntensity, 
    hs.StepTotal
FROM hourly_calories AS hc
LEFT JOIN hourly_intensity AS hi 
    ON hc.Id = hi.Id AND hc.ActivityHour = hi.ActivityHour
LEFT JOIN hourly_steps AS hs 
    ON hc.Id = hs.Id AND hc.ActivityHour = hs.ActivityHour;
"""

merged_df = pd.read_sql_query(query, conn)

print(merged_df)

conn.close()



#---------------------


def graph(id, stat):
    """
    Displays a graph of a statistic over time for an individual with id = id.
    Available stats: 'Calories', 'TotalIntensity', 'AverageIntensity', 'StepTotal'
    """
    if stat not in ["Calories", "TotalIntensity", "AverageIntensity", "StepTotal"]:
        print("Invalid stat. Choose from: 'Calories', 'TotalIntensity', 'AverageIntensity', 'StepTotal'")
        return

    # Filter data for the given Id
    user_data = merged_df[merged_df["Id"] == id].sort_values(by="ActivityHour")

    if user_data.empty:
        print(f"No data found for Id {id}")
        return

    # Plot
    plt.figure(figsize=(10, 5))
    plt.plot(user_data["ActivityHour"], user_data[stat], marker="o", linestyle="-", label=stat, color="b")

    plt.xlabel("Time")
    plt.ylabel(stat)
    plt.title(f"{stat} Over Time for Id {id}")
    plt.xticks(rotation=45)
    plt.legend()
    plt.grid(True)
    plt.show()


# Example usage
graph(1503960366, "TotalIntensity")


#---------------------


# Connect to SQLite database
conn = sqlite3.connect('fitbit_database.db')


# SQL query to calculate average calories per hour for each user
query1 = """
SELECT 
    Id, 
    AVG(Calories) AS AvgCaloriesPerHour
FROM hourly_calories
GROUP BY Id;
"""

# Read into a DataFrame
df_avg_calories = pd.read_sql_query(query1, conn)

query2 = """
SELECT 
    Id, 
    AVG(StepTotal) AS AvgStepsPerHour
FROM hourly_steps
GROUP BY Id;
"""

# Read into a DataFrame
df_avg_steps = pd.read_sql_query(query2, conn)

query3 = """
SELECT 
    Id, 
    SUM(value) AS Sleep
FROM minute_sleep
GROUP BY Id;
"""

# Read into a DataFrame
df_sleep = pd.read_sql_query(query3, conn)

# Fetch average intensity per user
query4 = """
SELECT 
    Id, 
    AVG(AverageIntensity) AS AvgIntensity
FROM hourly_intensity
GROUP BY Id;
"""
df_avg_intensity = pd.read_sql_query(query4, conn)



# Merge all DataFrames on 'Id'
df_merged = df_avg_calories.merge(df_avg_steps, on="Id", how="outer") \
                           .merge(df_sleep, on="Id", how="outer") \
                           .merge(df_avg_intensity, on="Id", how="outer")

df_merged["Sleep"] = df_merged["Sleep"].fillna(0)


query5 = """
SELECT * FROM weight_log;
"""
df_weight_log = pd.read_sql_query(query5, conn)

# Keep only the most recent weight entry per user
df_weight_log_sorted = df_weight_log.sort_values(by=["Id", "Date"], ascending=[True, False])
df_latest_weight = df_weight_log_sorted.drop_duplicates(subset=["Id"], keep="first")

# Merge on Id
df_merged2 = df_latest_weight.merge(df_merged, on="Id", how="left")

query6 = """
SELECT 
    Id,
    AVG(TotalSteps) AS AvgTotalSteps,
    AVG(TotalDistance) AS AvgTotalDistance,
    AVG(TrackerDistance) AS AvgTrackerDistance,
    AVG(LoggedActivitiesDistance) AS AvgLoggedActivitiesDistance,
    AVG(VeryActiveDistance) AS AvgVeryActiveDistance,
    AVG(ModeratelyActiveDistance) AS AvgModeratelyActiveDistance,
    AVG(LightActiveDistance) AS AvgLightActiveDistance,
    AVG(SedentaryActiveDistance) AS AvgSedentaryActiveDistance,
    AVG(VeryActiveMinutes) AS AvgVeryActiveMinutes,
    AVG(FairlyActiveMinutes) AS AvgFairlyActiveMinutes,
    AVG(LightlyActiveMinutes) AS AvgLightlyActiveMinutes,
    AVG(SedentaryMinutes) AS AvgSedentaryMinutes
FROM daily_activity
GROUP BY Id;
"""
df_avg_activity = pd.read_sql_query(query6, conn)

# Close the connection
conn.close()

# Merge with the existing merged dataset
df_avg_final = df_merged2.merge(df_avg_activity, on="Id", how="outer")



# Close the database connection
conn.close()


# Display the merged DataFrame
print(df_avg_final)


#---------------------



# Perform linear regression, change names in X and y to change the two compared values
df_final = df_avg_final.dropna(subset=["BMI", "Sleep"]) #If necessary
X = df_final[['BMI']] 
y = df_final['Sleep']

model = LinearRegression()
model.fit(X, y)

print(f"\nIntercept: {model.intercept_}")
print(f"Coefficient: {model.coef_[0]}")
print(f"R-squared: {model.score(X, y)}")

# Calculate residuals for qqplot
residuals = y - model.predict(X)

plt.figure(figsize=(12, 5))

# Scatter plot with regression line
plt.subplot(1, 2, 1)
plt.scatter(X, y, color='blue', label='Data points')
plt.plot(X, model.predict(X), color='red', label='Regression line')
plt.xlabel('BMI')
plt.ylabel('Total Sleep')
plt.title('Regression of Sleep Duration on BMI')
plt.legend()

# Q-Q plot for residuals
plt.subplot(1, 2, 2)
stats.probplot(residuals, dist="norm", plot=plt)
plt.title('Q-Q Plot of Residuals')
plt.xlabel('Theoretical Quantiles')
plt.ylabel('Sample Quantiles')

plt.tight_layout()
plt.show(block=False)


#-------------------

def get_activity_summary(conn, user_id, date):
    """
    Fetches Calories, TotalSteps, and Total Active Minutes (Very + Fairly + Lightly) 
    for a given user on a specific date.
    
    Parameters:
        conn (sqlite3.Connection): SQLite database connection
        user_id (int): The Id of the user
        date (str): The date in 'YYYY-MM-DD' format
    
    Returns:
        dict: A dictionary with Calories, TotalSteps, and TotalActiveMinutes
    """
    query = """
    SELECT Calories, TotalSteps, 
           (VeryActiveMinutes + FairlyActiveMinutes + LightlyActiveMinutes) AS TotalActiveMinutes
    FROM daily_activity
    WHERE Id = ? AND ActivityDate = ?;
    """
    
    df = pd.read_sql_query(query, conn, params=(user_id, date))
    
    if df.empty:
        return f"No data found for Id {user_id} on {date}"
    
    return df.iloc[0].to_dict()  # Convert row to dictionary

# Example usage
conn = sqlite3.connect('fitbit_database.db')
result = get_activity_summary(conn, 1503960366, "4/5/2016")
conn.close()

print(result)


#--------------------

def get_total_summary(conn, user_id):
    """
    Fetches Calories, TotalSteps, and Total Active Minutes (Very + Fairly + Lightly) 
    for a given user on a specific date.
    
    Parameters:
        conn (sqlite3.Connection): SQLite database connection
        user_id (int): The Id of the user
        date (str): The date in 'YYYY-MM-DD' format
    
    Returns:
        dict: A dictionary with Calories, TotalSteps, and TotalActiveMinutes
    """
    query = """
    SELECT SUM(Calories) AS Calories, SUM(TotalSteps) AS TotalSteps, 
           SUM(VeryActiveMinutes + FairlyActiveMinutes + LightlyActiveMinutes) AS TotalActiveMinutes
    FROM daily_activity
    WHERE Id = ?;
    """
    
    df = pd.read_sql_query(query, conn, params=(user_id,))
    
    if df.empty:
        return f"No data found for Id {user_id}"
    
    return df.iloc[0].to_dict()  # Convert row to dictionary

# Example usage
conn = sqlite3.connect('fitbit_database.db')
result = get_total_summary(conn, 1503960366)
conn.close()

print(result)
