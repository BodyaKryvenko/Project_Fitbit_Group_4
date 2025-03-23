import streamlit as st
import pandas as pd
import sqlite3

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

    if page == "Home":
        pass
    elif page == "User Analysis":
        pass
    elif page == "Time Analysis":
        pass
    elif page == "Sleep Analysis":
        pass
