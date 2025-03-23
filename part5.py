import streamlit as st
import pandas as pd
import sqlite3
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import statsmodels.api as sm
import numpy as np

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

    hourly_steps = pd.read_sql_query("SELECT * FROM hourly_steps", conn)
    hourly_calories = pd.read_sql_query("SELECT * FROM hourly_calories", conn)
    hourly_intensity = pd.read_sql_query("SELECT * FROM hourly_intensity", conn)
    daily_activity = pd.read_sql_query("SELECT * FROM daily_activity", conn)

    cursor = conn.cursor()
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='minute_sleep'")
    if cursor.fetchone():
        sleep_data = pd.read_sql_query("SELECT * FROM minute_sleep", conn)
    else:
        sleep_data = pd.DataFrame()

    cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='weight_log'")
    if cursor.fetchone():
        weight_data = pd.read_sql_query("SELECT * FROM weight_log", conn)
    else:
        weight_data = pd.DataFrame()

    conn.close()

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
    user_activity_counts = df['Id'].value_counts().reset_index()
    user_activity_counts.columns = ['Id', 'ActivityCount']
    user_activity_counts['Class'] = user_activity_counts['ActivityCount'].apply(classify_user)
    return user_activity_counts


def plot_total_distance_per_user(df):
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
    df['DayOfWeek'] = df['ActivityDate'].dt.day_name()

    day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']

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

    user_data = daily_activity[daily_activity['Id'] == user_id].copy()

    if user_data.empty:
        return None

    if selected_date:
        user_data = user_data[user_data['ActivityDate'].dt.date == selected_date]

    if user_data.empty:
        return None

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
    user_data = df[df['Id'] == user_id].copy()

    if start_date:
        start_date = pd.to_datetime(start_date)
        user_data = user_data[user_data['ActivityDate'] >= start_date]
    if end_date:
        end_date = pd.to_datetime(end_date)
        user_data = user_data[user_data['ActivityDate'] <= end_date]

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

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(user_data['TotalSteps'], user_data['Calories'], alpha=0.7)

    X = user_data['TotalSteps']
    X_with_const = sm.add_constant(X)
    y = user_data['Calories']

    model = sm.OLS(y, X_with_const).fit()

    x_range = np.linspace(X.min(), X.max(), 100)
    X_pred = sm.add_constant(x_range)
    y_pred = model.predict(X_pred)

    ax.plot(x_range, y_pred, 'r-', linewidth=2)

    ax.set_title(f'Relationship between Steps and Calories for User {user_id}')
    ax.set_xlabel('Total Steps')
    ax.set_ylabel('Calories Burnt')
    ax.grid(True, alpha=0.3)

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

    very_active = user_data['VeryActiveMinutes'].mean()
    fairly_active = user_data['FairlyActiveMinutes'].mean()
    lightly_active = user_data['LightlyActiveMinutes'].mean()
    sedentary = user_data['SedentaryMinutes'].mean()

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

    user_data = hourly_data[hourly_data['Id'] == user_id].copy()

    if user_data.empty:
        return None

    user_data['ActivityHour'] = pd.to_datetime(user_data['ActivityHour'])

    if start_date:
        start_date = pd.to_datetime(start_date)
        user_data = user_data[user_data['ActivityHour'].dt.date >= start_date.date()]
    if end_date:
        end_date = pd.to_datetime(end_date)
        user_data = user_data[user_data['ActivityHour'].dt.date <= end_date.date()]

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(user_data['ActivityHour'], user_data[y_column], marker='o', linestyle='-')
    ax.set_xlabel('Time')
    ax.set_ylabel(metric)
    ax.set_title(f'{title} for User {user_id}')
    plt.xticks(rotation=45)
    plt.grid(True)
    plt.tight_layout()

    return fig


def main():
    activity_data_file_path = "daily_activity.csv"
    db_file_path = "fitbit_database.db"

    activity_df = load_activity_data(activity_data_file_path)
    db_data = load_database_data(db_file_path)

    if not activity_df.empty:
        activity_df = preprocess_time_data(activity_df)

    for key in db_data:
        if not db_data[key].empty:
            db_data[key] = preprocess_time_data(db_data[key])

    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Go to", ["Home", "User Analysis", "Time Analysis", "Sleep Analysis"])

    if not activity_df.empty:
        user_ids = sorted(activity_df['Id'].unique())
    else:
        user_ids = []

    if user_ids:
        selected_user_id = st.sidebar.selectbox("Select User ID", user_ids)
    else:
        selected_user_id = None

    if page == "Home":
        st.title("Fitbit User Analysis Dashboard")

        st.write("""
                        This dashboard provides analysis of Fitbit user data, including activity patterns, 
                        sleep duration, and fitness metrics. Use the sidebar to navigate through different 
                        sections and select specific users for detailed analysis.
                        """)

        st.header("Study Overview")

        if not activity_df.empty:
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
            if not activity_df.empty:
                st.subheader("Total Distance per User")
                fig = plot_total_distance_per_user(activity_df)
                st.pyplot(fig)

        with col2:
            if not activity_df.empty:
                st.subheader("Workout Frequency by Day")
                fig = plot_workout_frequency_by_day(activity_df)
                st.pyplot(fig)

        st.header("User Classification")

        if not activity_df.empty:
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
    elif page == "User Analysis":
        st.title("User Activity Analysis")

        if selected_user_id:
            st.header(f"User {selected_user_id} Analysis")

            col1, col2 = st.columns(2)

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

            with col1:
                start_date = st.date_input("Start Date", min_date, min_value=min_date, max_value=max_date)

            with col2:
                end_date = st.date_input("End Date", max_date, min_value=min_date, max_value=max_date)

            if not activity_df.empty:
                activity_data = activity_df
            elif 'daily_activity' in db_data and not db_data['daily_activity'].empty:
                activity_data = db_data['daily_activity']
            else:
                activity_data = pd.DataFrame()

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

            st.subheader("User Activity Visualizations")

            tab1, tab2, tab3 = st.tabs(["Calories", "Steps-Calories Relationship", "Activity Breakdown"])

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
    elif page == "Time Analysis":
        pass
    elif page == "Sleep Analysis":
        pass
