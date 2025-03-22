import sqlite3

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

---------------------#
import sqlite3

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


---------------------#


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



---------------------#


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

