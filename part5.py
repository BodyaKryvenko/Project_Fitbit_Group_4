import streamlit as st
import pandas as pd
import sqlite3
import matplotlib.pyplot as plt

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
        pass
    elif page == "Time Analysis":
        pass
    elif page == "Sleep Analysis":
        pass
