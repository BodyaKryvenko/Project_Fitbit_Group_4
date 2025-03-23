import streamlit as st
import pandas as pd
import sqlite3
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import statsmodels.api as sm
import numpy as np
from sklearn.linear_model import LinearRegression

# Configure the Streamlit page
st.set_page_config(
    page_title="Fitbit User Analysis Dashboard",
    layout="wide",
    initial_sidebar_state="expanded"
)


def load_activity_data(file_path):
    activity_df = pd.read_csv(file_path)
    return activity_df


def load_database_data(file_path):
    conn = sqlite3.connect(file_path)

    # Reading tables from the database
    hourly_steps = pd.read_sql_query("SELECT * FROM hourly_steps", conn)
    hourly_calories = pd.read_sql_query("SELECT * FROM hourly_calories", conn)
    hourly_intensity = pd.read_sql_query("SELECT * FROM hourly_intensity", conn)
    daily_activity = pd.read_sql_query("SELECT * FROM daily_activity", conn)

    # Checking if minute_sleep table exists; read it if present
    cursor = conn.cursor()
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='minute_sleep'")
    if cursor.fetchone():
        sleep_data = pd.read_sql_query("SELECT * FROM minute_sleep", conn)
    else:
        sleep_data = pd.DataFrame()

    # Checking if weight_log table exists; read it if present
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='weight_log'")
    if cursor.fetchone():
        weight_data = pd.read_sql_query("SELECT * FROM weight_log", conn)
    else:
        weight_data = pd.DataFrame()

    conn.close()

    # Return all data as a dictionary
    data = {
        'hourly_steps': hourly_steps,
        'hourly_calories': hourly_calories,
        'hourly_intensity': hourly_intensity,
        'daily_activity': daily_activity,
        'sleep_data': sleep_data,
        'weight_data': weight_data
    }

    return data


def preprocess_time_data(df):
    # Converting any column that could represent a date/time to datetime
    if 'ActivityDate' in df.columns:
        df['ActivityDate'] = pd.to_datetime(df['ActivityDate'])

    if 'ActivityHour' in df.columns:
        df['ActivityHour'] = pd.to_datetime(df['ActivityHour'])

    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'])

    if 'Date' in df.columns:
        df['Date'] = pd.to_datetime(df['Date'])

    return df


def classify_user(activity_count):
    if activity_count <= 10:
        return 'Light user'
    elif 11 <= activity_count <= 15:
        return 'Moderate user'
    else:
        return 'Heavy user'


def get_user_classification(df):
    # Counting how many rows of activity data each user has
    user_activity_counts = df['Id'].value_counts().reset_index()
    user_activity_counts.columns = ['Id', 'ActivityCount']

    # Apply classification function
    user_activity_counts['Class'] = user_activity_counts['ActivityCount'].apply(classify_user)
    return user_activity_counts


def plot_total_distance_per_user(df):
    # Sum the total distance for each user
    total_distance_per_user = df.groupby('Id')['TotalDistance'].sum().reset_index()
    total_distance_per_user = total_distance_per_user.sort_values('TotalDistance', ascending=False)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(total_distance_per_user['Id'].astype(str), total_distance_per_user['TotalDistance'])
    ax.set_xlabel('User ID')
    ax.set_ylabel('Total Distance (km)')
    ax.set_title('Total Distance per User')
    plt.xticks(rotation=90)
    plt.tight_layout()

    return fig


def plot_workout_frequency_by_day(df):
    # Extracting the day name from the ActivityDate
    df['DayOfWeek'] = df['ActivityDate'].dt.day_name()

    # Defining a custom day order
    day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']

    # Counting how many activity entries fall on each day of the week
    workout_frequency = df['DayOfWeek'].value_counts().reindex(day_order).fillna(0)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(workout_frequency.index, workout_frequency.values)
    ax.set_xlabel('Day of the Week')
    ax.set_ylabel('Number of Workouts')
    ax.set_title('Frequency of Workouts by Day of the Week')
    plt.xticks(rotation=45)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    return fig


def get_activity_summary(daily_activity, user_id, selected_date=None):
    if daily_activity.empty:
        return None

    # Filtering the DataFrame for the specified user
    user_data = daily_activity[daily_activity['Id'] == user_id].copy()

    if user_data.empty:
        return None

    # Further filtering by selected_date if provided
    if selected_date:
        user_data = user_data[user_data['ActivityDate'].dt.date == selected_date]

    if user_data.empty:
        return None

    # If a single date is selected, return specific metrics for that date
    if selected_date:
        row = user_data.iloc[0]
        return {
            'Calories': row['Calories'],
            'TotalSteps': row['TotalSteps'],
            'TotalDistance': row['TotalDistance'],
            'VeryActiveMinutes': row['VeryActiveMinutes'],
            'FairlyActiveMinutes': row['FairlyActiveMinutes'],
            'LightlyActiveMinutes': row['LightlyActiveMinutes'],
            'SedentaryMinutes': row['SedentaryMinutes'],
            'TotalActiveMinutes': row['VeryActiveMinutes'] + row['FairlyActiveMinutes'] + row['LightlyActiveMinutes']
        }

    # Otherwise, return mean values across all days for this user
    return {
        'Calories': user_data['Calories'].mean(),
        'TotalSteps': user_data['TotalSteps'].mean(),
        'TotalDistance': user_data['TotalDistance'].mean(),
        'VeryActiveMinutes': user_data['VeryActiveMinutes'].mean(),
        'FairlyActiveMinutes': user_data['FairlyActiveMinutes'].mean(),
        'LightlyActiveMinutes': user_data['LightlyActiveMinutes'].mean(),
        'SedentaryMinutes': user_data['SedentaryMinutes'].mean(),
        'TotalActiveMinutes': (user_data['VeryActiveMinutes'] + user_data['FairlyActiveMinutes'] + user_data[
            'LightlyActiveMinutes']).mean()
    }


def plot_calories_burnt(df, user_id, start_date=None, end_date=None):
    # Filtering the data for the given user
    user_data = df[df['Id'] == user_id].copy()

    # Filtering by start_date if provided
    if start_date:
        start_date = pd.to_datetime(start_date)
        user_data = user_data[user_data['ActivityDate'] >= start_date]

    # Filtering by end_date if provided
    if end_date:
        end_date = pd.to_datetime(end_date)
        user_data = user_data[user_data['ActivityDate'] <= end_date]

    # Creating the plot
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(user_data['ActivityDate'], user_data['Calories'], marker='o', linestyle='-')
    ax.set_xlabel('Date')
    ax.set_ylabel('Calories Burnt')
    ax.set_title(f'Calories Burnt per Day for User {user_id}')
    plt.xticks(rotation=45)
    plt.grid(True)
    plt.tight_layout()

    return fig


def plot_steps_calories_relationship(df, user_id):
    user_data = df[df['Id'] == user_id].copy()

    if user_data.empty:
        return None

    # Creating a scatter plot of TotalSteps vs. Calories
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(user_data['TotalSteps'], user_data['Calories'], alpha=0.7)

    # Building a simple linear regression model
    X = user_data['TotalSteps']
    X_with_const = sm.add_constant(X)
    y = user_data['Calories']
    model = sm.OLS(y, X_with_const).fit()

    # Predicting values for the regression line
    x_range = np.linspace(X.min(), X.max(), 100)
    X_pred = sm.add_constant(x_range)
    y_pred = model.predict(X_pred)

    # Plot the regression line
    ax.plot(x_range, y_pred, 'r-', linewidth=2)

    # Setting labels and title
    ax.set_title(f'Relationship between Steps and Calories for User {user_id}')
    ax.set_xlabel('Total Steps')
    ax.set_ylabel('Calories Burnt')
    ax.grid(True, alpha=0.3)

    # Adding regression equation and R-squared to the plot
    equation = f"Calories = {model.params[0]:.2f} + {model.params[1]:.4f} * Steps"
    r_squared = f"R² = {model.rsquared:.3f}"
    ax.annotate(equation + "\n" + r_squared,
                xy=(0.05, 0.95),
                xycoords='axes fraction',
                fontsize=10,
                bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8))

    plt.tight_layout()

    return fig


def plot_activity_breakdown(user_data):
    if user_data.empty:
        return None

    # Calculating average minutes in each activity level
    very_active = user_data['VeryActiveMinutes'].mean()
    fairly_active = user_data['FairlyActiveMinutes'].mean()
    lightly_active = user_data['LightlyActiveMinutes'].mean()
    sedentary = user_data['SedentaryMinutes'].mean()

    # Creating pie chart
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.pie(
        [very_active, fairly_active, lightly_active, sedentary],
        labels=['Very Active', 'Fairly Active', 'Lightly Active', 'Sedentary'],
        autopct='%1.1f%%',
        startangle=90,
        colors=['#28a745', '#17a2b8', '#ffc107', '#dc3545']
    )
    ax.set_title('Average Daily Activity Breakdown')
    ax.axis('equal')
    plt.tight_layout()

    return fig


def plot_activity_over_time(db_data, user_id, metric, start_date=None, end_date=None):
    hourly_data = None

    # Selecting the appropriate table and columns based on metric
    if metric == 'Steps':
        hourly_data = db_data['hourly_steps'].copy()
        y_column = 'StepTotal'
        title = 'Steps Over Time'
    elif metric == 'Calories':
        hourly_data = db_data['hourly_calories'].copy()
        y_column = 'Calories'
        title = 'Calories Over Time'
    elif metric == 'Intensity':
        hourly_data = db_data['hourly_intensity'].copy()
        y_column = 'TotalIntensity'
        title = 'Activity Intensity Over Time'

    if hourly_data is None or hourly_data.empty:
        return None

    # Filtering for the specified user
    user_data = hourly_data[hourly_data['Id'] == user_id].copy()

    if user_data.empty:
        return None

    # Converting to datetime
    user_data['ActivityHour'] = pd.to_datetime(user_data['ActivityHour'])

    # Filtering by date range if provided
    if start_date:
        start_date = pd.to_datetime(start_date)
        user_data = user_data[user_data['ActivityHour'].dt.date >= start_date.date()]

    # Filtering by end_date if provided
    if end_date:
        end_date = pd.to_datetime(end_date)
        user_data = user_data[user_data['ActivityHour'].dt.date <= end_date.date()]

    # Creating figure
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(user_data['ActivityHour'], user_data[y_column], marker='o', linestyle='-')
    ax.set_xlabel('Time')
    ax.set_ylabel(metric)
    ax.set_title(f'{title} for User {user_id}')
    plt.xticks(rotation=45)
    plt.grid(True)
    plt.tight_layout()

    return fig


def plot_time_of_day_analysis(db_data, metric):
    if not db_data:
        return None

    # Define the 4-hour blocks in chronological order
    time_blocks = {
        '0-4': (0, 4),
        '4-8': (4, 8),
        '8-12': (8, 12),
        '12-16': (12, 16),
        '16-20': (16, 20),
        '20-24': (20, 24)
    }

    chronological_order = ['0-4', '4-8', '8-12', '12-16', '16-20', '20-24']

    # Helper function to assign a block based on the hour
    def assign_time_block(hour):
        for block, (start, end) in time_blocks.items():
            if start <= hour < end:
                return block
        return None

    if metric == 'Steps':
        hourly_data = db_data['hourly_steps'].copy()
        hourly_data['ActivityHour'] = pd.to_datetime(hourly_data['ActivityHour'])
        hourly_data['Hour'] = hourly_data['ActivityHour'].dt.hour
        hourly_data['TimeBlock'] = hourly_data['Hour'].apply(assign_time_block)

        # Calculating average steps per time block
        average_per_block = hourly_data.groupby('TimeBlock')['StepTotal'].mean().reset_index()

        y_column = 'StepTotal'
        title = 'Average Steps per 4-Hour Time Block'
        y_label = 'Average Steps'

    elif metric == 'Calories':
        hourly_data = db_data['hourly_calories'].copy()
        hourly_data['ActivityHour'] = pd.to_datetime(hourly_data['ActivityHour'])
        hourly_data['Hour'] = hourly_data['ActivityHour'].dt.hour
        hourly_data['TimeBlock'] = hourly_data['Hour'].apply(assign_time_block)

        # Calculating average calories per time block
        average_per_block = hourly_data.groupby('TimeBlock')['Calories'].mean().reset_index()

        y_column = 'Calories'
        title = 'Average Calories Burnt per 4-Hour Time Block'
        y_label = 'Average Calories'

    elif metric == 'Intensity':
        hourly_data = db_data['hourly_intensity'].copy()
        hourly_data['ActivityHour'] = pd.to_datetime(hourly_data['ActivityHour'])
        hourly_data['Hour'] = hourly_data['ActivityHour'].dt.hour
        hourly_data['TimeBlock'] = hourly_data['Hour'].apply(assign_time_block)

        # Calculating average intensity per time block
        average_per_block = hourly_data.groupby('TimeBlock')['TotalIntensity'].mean().reset_index()

        y_column = 'TotalIntensity'
        title = 'Average Intensity per 4-Hour Time Block'
        y_label = 'Average Intensity'

    elif metric == 'Sleep':
        sleep_data = db_data.get('sleep_data', pd.DataFrame())

        if sleep_data.empty:
            return None

        sleep_data['date'] = pd.to_datetime(sleep_data['date'])
        sleep_data['Hour'] = sleep_data['date'].dt.hour
        sleep_data['TimeBlock'] = sleep_data['Hour'].apply(assign_time_block)

        # Filtering for sleep data
        sleep_data = sleep_data[sleep_data['value'] == 1]

        # Counting sleep minutes per time block
        sleep_minutes = sleep_data.groupby('TimeBlock').size().reset_index(name='SleepMinutes')
        sleep_minutes['SleepMinutes'] /= len(sleep_data['Id'].unique())

        average_per_block = sleep_minutes

        y_column = 'SleepMinutes'
        title = 'Average Sleep Minutes per 4-Hour Time Block'
        y_label = 'Average Sleep Minutes'

    # Converting to ordered category for proper sorting
    average_per_block['TimeBlock'] = pd.Categorical(
        average_per_block['TimeBlock'],
        categories=chronological_order,
        ordered=True
    )
    average_per_block = average_per_block.sort_values('TimeBlock')

    # Create figure
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(average_per_block['TimeBlock'], average_per_block[y_column])
    ax.set_xlabel('Time Block')
    ax.set_ylabel(y_label)
    ax.set_title(title)
    plt.tight_layout()

    return fig


def plot_sleep_analysis(db_data, analysis_type):
    sleep_data = db_data.get('sleep_data', pd.DataFrame())
    daily_activity = db_data.get('daily_activity', pd.DataFrame())

    # Checking if either table is missing or empty
    if sleep_data.empty or daily_activity.empty:
        return None

    # Converting columns to datetime
    sleep_data['date'] = pd.to_datetime(sleep_data['date'])
    daily_activity['ActivityDate'] = pd.to_datetime(daily_activity['ActivityDate'])

    # Filtering rows where value = 1 (indicating sleep)
    sleep_df = sleep_data[sleep_data['value'] == 1]

    # Calculating sleep duration in minutes per day per user
    sleep_duration_df = sleep_df.groupby(['Id', sleep_df['date'].dt.date]).size().reset_index(
        name='SleepDurationMinutes')
    sleep_duration_df['date'] = pd.to_datetime(sleep_duration_df['date'])

    # Merge with activity data
    merged_df = pd.merge(
        daily_activity,
        sleep_duration_df,
        left_on=['Id', daily_activity['ActivityDate'].dt.date],
        right_on=['Id', sleep_duration_df['date'].dt.date],
        how='inner'
    )

    if merged_df.empty:
        return None

    # Setting up the x-axis variable based on the analysis type
    if analysis_type == 'active_minutes':
        merged_df['TotalActiveMinutes'] = merged_df['VeryActiveMinutes'] + merged_df['FairlyActiveMinutes'] + merged_df[
            'LightlyActiveMinutes']
        x_column = 'TotalActiveMinutes'
        x_label = 'Total Active Minutes'
    elif analysis_type == 'sedentary_minutes':
        x_column = 'SedentaryMinutes'
        x_label = 'Sedentary Minutes'
    elif analysis_type == 'steps':
        x_column = 'TotalSteps'
        x_label = 'Total Steps'

    # Creating scatter plot
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(merged_df[x_column], merged_df['SleepDurationMinutes'], alpha=0.5)

    # Adding regression line
    X = merged_df[[x_column]]
    y = merged_df['SleepDurationMinutes']

    model = LinearRegression()
    model.fit(X, y)

    # Getting predictions for the regression line
    x_range = np.linspace(merged_df[x_column].min(), merged_df[x_column].max(), 100)
    x_range = x_range.reshape(-1, 1)
    y_pred = model.predict(x_range)

    # Plot the regression line
    ax.plot(x_range, y_pred, 'r-', linewidth=2)

    # Adding labels and title
    ax.set_title(f'Relationship between {x_label} and Sleep Duration')
    ax.set_xlabel(x_label)
    ax.set_ylabel('Sleep Duration (minutes)')
    ax.grid(True, alpha=0.3)

    # Adding equation and R-squared
    r_squared = model.score(X, y)
    coefficient = model.coef_[0]
    intercept = model.intercept_

    equation = f"Sleep = {intercept:.2f} + {coefficient:.4f} * {x_label}"
    r_squared_text = f"R² = {r_squared:.3f}"

    ax.annotate(equation + "\n" + r_squared_text,
                xy=(0.05, 0.95),
                xycoords='axes fraction',
                fontsize=10,
                bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8))

    plt.tight_layout()

    return fig


def plot_user_sleep_duration(sleep_data, user_id):
    # Filtering for the selected user
    user_sleep = sleep_data[sleep_data['Id'] == user_id].copy()

    if user_sleep.empty:
        return None

    # Filtering for sleep (value = 1)
    user_sleep = user_sleep[user_sleep['value'] == 1]

    # Grouping by date to get sleep duration per day
    user_sleep_duration = user_sleep.groupby(user_sleep['date'].dt.date).size().reset_index(name='SleepMinutes')
    user_sleep_duration['date'] = pd.to_datetime(user_sleep_duration['date'])

    # Plot sleep duration over time
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(user_sleep_duration['date'], user_sleep_duration['SleepMinutes'], marker='o', linestyle='-')
    ax.set_xlabel('Date')
    ax.set_ylabel('Sleep Duration (minutes)')
    ax.set_title(f'Sleep Duration for User {user_id}')

    # Adding 8 hours reference line
    ax.axhline(y=480, color='r', linestyle='--', label='8 hours')
    ax.legend()

    plt.xticks(rotation=45)
    plt.grid(True)
    plt.tight_layout()

    return fig, user_sleep_duration


def main():
    # File paths for CSV and SQLite database (modify them as needed)
    activity_data_file_path = "daily_acivity.csv"
    db_file_path = "fitbit_database.db"

    # Loading the CSV and database data
    activity_df = load_activity_data(activity_data_file_path)
    db_data = load_database_data(db_file_path)

    # Preprocessing the loaded DataFrames
    if not activity_df.empty:
        activity_df = preprocess_time_data(activity_df)

    for key in db_data:
        if not db_data[key].empty:
            db_data[key] = preprocess_time_data(db_data[key])

    # Setting up Streamlit Sidebar navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Go to", ["Home", "User Analysis", "Time Analysis", "Sleep Analysis"])

    # Gathering available user IDs for selection
    if not activity_df.empty:
        user_ids = sorted(activity_df['Id'].unique())
    else:
        user_ids = []

    if user_ids:
        selected_user_id = st.sidebar.selectbox("Select User ID", user_ids)
    else:
        selected_user_id = None

    # Home Page
    if page == "Home":
        st.title("Fitbit User Analysis Dashboard")

        st.write("""
                This dashboard provides analysis of Fitbit user data, including activity patterns, 
                sleep duration, and fitness metrics. Use the sidebar to navigate through different 
                sections and select specific users for detailed analysis.
                """)

        st.header("Study Overview")

        if not activity_df.empty:
            # Show overall summary metrics if activity_df is not empty
            col1, col2, col3, col4 = st.columns(4)

            with col1:
                st.metric("Participants", activity_df['Id'].nunique())

            with col2:
                avg_steps = int(activity_df['TotalSteps'].mean())
                st.metric("Avg Daily Steps", f"{avg_steps:,}")

            with col3:
                avg_calories = int(activity_df['Calories'].mean())
                st.metric("Avg Daily Calories", f"{avg_calories:,}")

            with col4:
                total_distance = int(activity_df['TotalDistance'].sum())
                st.metric("Total Distance (km)", f"{total_distance:,}")

        st.header("Graphical Summaries")

        col1, col2 = st.columns(2)

        with col1:
            # Plotting total distance per user if we have activity data
            if not activity_df.empty:
                st.subheader("Total Distance per User")
                fig = plot_total_distance_per_user(activity_df)
                st.pyplot(fig)

        with col2:
            # Plotting workout frequency by day if activity data is available
            if not activity_df.empty:
                st.subheader("Workout Frequency by Day")
                fig = plot_workout_frequency_by_day(activity_df)
                st.pyplot(fig)

        st.header("User Classification")

        if not activity_df.empty:
            # Show classification pie chart and data if activity data is available
            user_classification = get_user_classification(activity_df)

            class_counts = user_classification['Class'].value_counts().reset_index()
            class_counts.columns = ['Class', 'Count']

            col1, col2 = st.columns(2)

            with col1:
                fig, ax = plt.subplots(figsize=(8, 8))
                ax.pie(class_counts['Count'], labels=class_counts['Class'], autopct='%1.1f%%', startangle=90,
                       colors=['#91EE91', '#5DADE2', '#AF7AC5'])
                ax.axis('equal')
                ax.set_title('Distribution of User Types')
                st.pyplot(fig)

            with col2:
                st.dataframe(user_classification.sort_values('ActivityCount', ascending=False))

    # User Analysis Page
    elif page == "User Analysis":
        st.title("User Activity Analysis")

        if selected_user_id:
            st.header(f"User {selected_user_id} Analysis")

            col1, col2 = st.columns(2)

            # Determine date range from user data (either CSV or DB)
            if not activity_df.empty:
                user_data = activity_df[activity_df['Id'] == selected_user_id]
                min_date = user_data['ActivityDate'].min().date()
                max_date = user_data['ActivityDate'].max().date()
            elif 'daily_activity' in db_data and not db_data['daily_activity'].empty:
                user_data = db_data['daily_activity'][db_data['daily_activity']['Id'] == selected_user_id]
                min_date = user_data['ActivityDate'].min().date()
                max_date = user_data['ActivityDate'].max().date()
            else:
                min_date = datetime.now().date() - timedelta(days=30)
                max_date = datetime.now().date()

            # Letting user pick a date range
            with col1:
                start_date = st.date_input("Start Date", min_date, min_value=min_date, max_value=max_date)

            with col2:
                end_date = st.date_input("End Date", max_date, min_value=min_date, max_value=max_date)

            # Getting the relevant activity data from either CSV or DB
            if not activity_df.empty:
                activity_data = activity_df
            elif 'daily_activity' in db_data and not db_data['daily_activity'].empty:
                activity_data = db_data['daily_activity']
            else:
                activity_data = pd.DataFrame()

            # Show an overall summary if there's data
            if not activity_data.empty:
                summary = get_activity_summary(activity_data, selected_user_id)

                if summary:
                    st.subheader("Activity Summary")

                    col1, col2, col3, col4 = st.columns(4)

                    with col1:
                        st.metric("Daily Calories", f"{int(summary['Calories']):,}")

                    with col2:
                        st.metric("Daily Steps", f"{int(summary['TotalSteps']):,}")

                    with col3:
                        st.metric("Daily Distance (km)", f"{summary['TotalDistance']:.2f}")

                    with col4:
                        st.metric("Active Minutes", int(summary['TotalActiveMinutes']))

            # Tabs for different user-specific plots
            st.subheader("User Activity Visualizations")

            tab1, tab2, tab3 = st.tabs(["Calories", "Steps-Calories Relationship", "Activity Breakdown"])

            # Calories Over Time
            with tab1:
                if not activity_df.empty:
                    st.write("Calories Burnt Over Time")
                    calories_fig = plot_calories_burnt(activity_df, selected_user_id, start_date, end_date)
                    st.pyplot(calories_fig)
                elif 'daily_activity' in db_data and not db_data['daily_activity'].empty:
                    st.write("Calories Burnt Over Time")
                    calories_fig = plot_calories_burnt(db_data['daily_activity'], selected_user_id, start_date,
                                                       end_date)
                    st.pyplot(calories_fig)
                else:
                    st.warning("No calories data available for this user.")

            # Steps-Calories Relationship
            with tab2:
                if not activity_df.empty:
                    st.write("Relationship Between Steps and Calories")
                    relationship_fig = plot_steps_calories_relationship(activity_df, selected_user_id)
                    if relationship_fig:
                        st.pyplot(relationship_fig)
                    else:
                        st.warning("Insufficient data to plot steps-calories relationship.")
                elif 'daily_activity' in db_data and not db_data['daily_activity'].empty:
                    st.write("Relationship Between Steps and Calories")
                    relationship_fig = plot_steps_calories_relationship(db_data['daily_activity'], selected_user_id)
                    if relationship_fig:
                        st.pyplot(relationship_fig)
                    else:
                        st.warning("Insufficient data to plot steps-calories relationship.")
                else:
                    st.warning("No activity data available for this user.")

            # Activity Breakdown Pie Chart
            with tab3:
                if 'daily_activity' in db_data and not db_data['daily_activity'].empty:
                    user_data = db_data['daily_activity'][db_data['daily_activity']['Id'] == selected_user_id].copy()

                    if not user_data.empty:
                        user_data = user_data[(user_data['ActivityDate'].dt.date >= start_date) &
                                              (user_data['ActivityDate'].dt.date <= end_date)]

                        st.write("Average Daily Activity Breakdown")
                        activity_fig = plot_activity_breakdown(user_data)
                        if activity_fig:
                            st.pyplot(activity_fig)
                        else:
                            st.warning("Insufficient data to plot activity breakdown.")
                    else:
                        st.warning("No activity data available for this user in the selected date range.")
                elif not activity_df.empty:
                    st.warning(
                        "Activity breakdown not available in the CSV file. Use the database for more detailed analysis.")
                else:
                    st.warning("No activity data available for this user.")

            # Hourly activity patterns
            if db_data:
                st.subheader("Hourly Activity Patterns")

                metric = st.selectbox("Select Metric", ["Steps", "Calories", "Intensity"])

                hourly_fig = plot_activity_over_time(db_data, selected_user_id, metric, start_date, end_date)
                if hourly_fig:
                    st.pyplot(hourly_fig)
                else:
                    st.warning(f"No hourly {metric.lower()} data available for this user.")
        else:
            st.info("Please select a user ID from the sidebar to view detailed analysis.")

    # Time Analysis Page
    elif page == "Time Analysis":
        st.title("Time-Based Activity Analysis")

        st.write("""
            This section analyzes activity patterns across different times of day and days of the week.
            """)

        # Overall day-of-week patterns (using the CSV daily activity if available)
        if not activity_df.empty:
            st.header("Day of Week Patterns")
            day_fig = plot_workout_frequency_by_day(activity_df)
            st.pyplot(day_fig)

        # Time of day analysis (using database hourly data if available)
        if db_data:
            st.header("Time of Day Patterns")

            metric = st.selectbox("Select Metric", ["Steps", "Calories", "Intensity", "Sleep"])

            time_of_day_fig = plot_time_of_day_analysis(db_data, metric)
            if time_of_day_fig:
                st.pyplot(time_of_day_fig)
            else:
                st.warning(f"No {metric.lower()} data available for time of day analysis.")

        # User-specific time analysis
        if selected_user_id:
            st.header(f"Time Analysis for User {selected_user_id}")

            col1, col2 = st.columns(2)

            # Determine date range from user data
            if not activity_df.empty:
                user_data = activity_df[activity_df['Id'] == selected_user_id]
                min_date = user_data['ActivityDate'].min().date()
                max_date = user_data['ActivityDate'].max().date()
            elif 'daily_activity' in db_data and not db_data['daily_activity'].empty:
                user_data = db_data['daily_activity'][db_data['daily_activity']['Id'] == selected_user_id]
                min_date = user_data['ActivityDate'].min().date()
                max_date = user_data['ActivityDate'].max().date()
            else:
                min_date = datetime.now().date() - timedelta(days=30)
                max_date = datetime.now().date()

            # Letting user pick a date range
            with col1:
                start_date = st.date_input("Start Date", min_date, min_value=min_date, max_value=max_date,
                                           key="time_start_date")

            with col2:
                end_date = st.date_input("End Date", max_date, min_value=min_date, max_value=max_date,
                                         key="time_end_date")

            # Plot hourly data for the selected user
            if db_data:
                user_metric = st.selectbox("Select Metric", ["Steps", "Calories", "Intensity"], key="user_time_metric")

                hourly_fig = plot_activity_over_time(db_data, selected_user_id, user_metric, start_date, end_date)
                if hourly_fig:
                    st.pyplot(hourly_fig)
                else:
                    st.warning(
                        f"No hourly {user_metric.lower()} data available for this user in the selected date range.")

    # Sleep Analysis Page
    elif page == "Sleep Analysis":
        st.title("Sleep Duration Analysis")

        st.write("""
                This section analyzes factors affecting sleep duration based on activity patterns.
                """)

        # If sleep data is in the DB, let the user compare sleep vs. activity metrics
        if db_data and 'sleep_data' in db_data and not db_data['sleep_data'].empty:
            analysis_type = st.selectbox(
                "Analyze Sleep Duration vs:",
                ["active_minutes", "sedentary_minutes", "steps"],
                format_func=lambda x: {
                    "active_minutes": "Active Minutes",
                    "sedentary_minutes": "Sedentary Minutes",
                    "steps": "Total Steps"
                }[x]
            )

            # Plot scatter + regression line
            sleep_fig = plot_sleep_analysis(db_data, analysis_type)
            if sleep_fig:
                st.pyplot(sleep_fig)

            # Time of day sleep patterns
            st.header("Sleep Patterns by Time of Day")

            sleep_time_fig = plot_time_of_day_analysis(db_data, "Sleep")
            if sleep_time_fig:
                st.pyplot(sleep_time_fig)

            # User-specific sleep analysis
            if selected_user_id:
                st.header(f"Sleep Analysis for User {selected_user_id}")

                sleep_data = db_data['sleep_data']

                # Getting sleep duration over time for the selected user
                sleep_fig, user_sleep_duration = plot_user_sleep_duration(sleep_data, selected_user_id)

                if sleep_fig:
                    st.pyplot(sleep_fig)

                    # Calculating sleep statistics
                    avg_sleep = user_sleep_duration['SleepMinutes'].mean()
                    min_sleep = user_sleep_duration['SleepMinutes'].min()
                    max_sleep = user_sleep_duration['SleepMinutes'].max()

                    col1, col2, col3 = st.columns(3)

                    with col1:
                        st.metric("Avg Sleep (hours)", f"{avg_sleep / 60:.1f}")

                    with col2:
                        st.metric("Min Sleep (hours)", f"{min_sleep / 60:.1f}")

                    with col3:
                        st.metric("Max Sleep (hours)", f"{max_sleep / 60:.1f}")

                    # Comparing to activity
                    activity_data = db_data.get('daily_activity', pd.DataFrame())

                    if not activity_data.empty:
                        # Filtering for the selected user
                        user_activity = activity_data[activity_data['Id'] == selected_user_id].copy()

                        if not user_activity.empty:
                            # Merging sleep and activity data
                            merged_df = pd.merge(
                                user_sleep_duration,
                                user_activity,
                                left_on=user_sleep_duration['date'].dt.date,
                                right_on=user_activity['ActivityDate'].dt.date,
                                how='inner'
                            )

                            if not merged_df.empty:
                                st.subheader("Relationship Between Activity and Sleep")

                                # Creating scatter plot with steps and sleep
                                fig, ax = plt.subplots(figsize=(10, 6))
                                ax.scatter(merged_df['TotalSteps'], merged_df['SleepMinutes'], alpha=0.7)
                                ax.set_xlabel('Total Steps')
                                ax.set_ylabel('Sleep Duration (minutes)')
                                ax.set_title('Sleep Duration vs. Steps')
                                ax.grid(True, alpha=0.3)
                                plt.tight_layout()

                                st.pyplot(fig)
                else:
                    st.warning("No sleep data available for this user.")
        else:
            st.warning("Sleep data is not available. Please make sure the database contains the minute_sleep table.")


if __name__ == "__main__":
    main()
