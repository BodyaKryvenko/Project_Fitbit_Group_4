# Project Fitbit Group 4

# Fitbit Data Analysis

## Project Overview
The aim of this project is to analyse Fitbit data from 33 users who participated in a 2016 Amazon survey. We explored activity patterns, sleep habits, and other health metrics to gain insights into user behavior.

## Data Files
- `daily_acivity.csv`: Daily user activity data
- `fitbit_database.db`: Database with tables for heart rate, sleep, calories, steps, and more
- `chicago.csv`: Weather measuremens from the time and place where the survey was taken

## Repository Structure
- `part1.py`: Basic data exploration and visualization
- `part3.py`: User classification and sleep analysis
- `part4.py`: Weight data analysis and additional metrics
- `part5.py`: Interactive dashboard

## Getting Started

### Requirements
- Python 3.10+
- Required packages: pandas, matplotlib, numpy, statsmodels, sqlite3, scikit-learn, streamlit

## How to Use
Run the following commands in your terminal to execute the scripts.

### Run Basic Analysis
```
python part1.py
```
This shows user counts, distances, calorie patterns, and workout frequency.

### Run Database Analysis
```
python part3.py
```
This analyzes sleep patterns, activity by time of day, and heart rate data.

### Run Weight Analysis
```
python part4.py
```
This handles missing values and compares BMI with other metrics.

### Launch Dashboard
```
streamlit run part5.py
```
The dashboard lets you:
- View overall statistics
- Select specific users
- Compare user metrics
- Filter by date range
- Analyse sleep patterns
- See activity breakdowns
