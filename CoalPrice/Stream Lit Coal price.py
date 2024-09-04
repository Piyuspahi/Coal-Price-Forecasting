

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sqlalchemy import create_engine, text
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.ensemble import GradientBoostingRegressor
from statsmodels.tsa.stattools import adfuller

# Define function to load data
@st.cache_data
def load_data(file_path):
    data = pd.read_csv(file_path, parse_dates=['Date'], dayfirst=True)  # Adjusted date parsing
    return data

# Define function to preprocess data
def preprocess_data(data, target_column, train_start, train_end, test_start, test_end):
    data['Date'] = pd.to_datetime(data['Date'])  # Ensure 'Date' column is in datetime format
    data.set_index('Date', inplace=True)
    
    # Handle missing and infinite values
    data.replace([np.inf, -np.inf], np.nan, inplace=True)
    
    # Separate features and target
    X = data.drop(columns=[target_column])
    y = data[target_column]
    
    # Impute missing values
    imputer = SimpleImputer(strategy='mean')
    X_imputed = imputer.fit_transform(X)
    y_imputed = y.fillna(y.mean())  # Impute missing target values

    # Split data according to the given date ranges
    X_train = X_imputed[(data.index >= train_start) & (data.index <= train_end)]
    y_train = y_imputed[(data.index >= train_start) & (data.index <= train_end)]
    X_test = X_imputed[(data.index >= test_start) & (data.index <= test_end)]
    y_test = y_imputed[(data.index >= test_start) & (data.index <= test_end)]
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    return X_train_scaled, X_test_scaled, y_train, y_test, scaler

# Streamlit app
def main():
    st.title("Coal Price Forecasting")

    # HTML-based interface
    st.markdown("""
        <style>
            .main {background-color: #F5F5F5; padding: 20px; border-radius: 10px;}
            .title {font-size: 24px; font-weight: bold;}
            .subtitle {font-size: 20px; margin-top: 10px;}
            .text {font-size: 16px; margin-top: 5px;}
        </style>
        <div class="main">
            <div class="title">Coal Price Forecasting</div>
            <div class="subtitle">Upload your dataset</div>
        </div>
    """, unsafe_allow_html=True)

    # File upload
    uploaded_file = st.file_uploader("Choose a file", type="csv")
    if uploaded_file is not None:
        data = load_data(uploaded_file)
        st.write("Data loaded successfully!")

        # Display first 5 rows
        st.subheader("First 5 rows of the data")
        st.write(data.head())

        # Connect with MySQL
        user = "root"
        pw = "965877"
        db = "coal_db"
        engine = create_engine(f"mysql+pymysql://{user}:{pw}@localhost/{db}")

        data.to_sql("coal", con=engine, if_exists="replace", index=False)
        sql = "select * from coal;"
        coal = pd.read_sql_query(text(sql), engine.connect())
        st.write("Data loaded from MySQL")
        st.write(coal.head(10))

        # Exploratory Data Analysis
        st.subheader("Exploratory Data Analysis")
        columns = [
            'Coal_RB_4800_FOB_London_Close_USD', 'Coal_RB_5500_FOB_London_Close_USD',
            'Coal_RB_5700_FOB_London_Close_USD', 'Coal_RB_6000_FOB_CurrentWeek_Avg_USD',
            'Coal_India_5500_CFR_London_Close_USD'
        ]

        # Display summary statistics
        st.write("Summary Statistics")
        st.write(data.describe())

        # Display correlation heatmap
        st.write("Correlation Heatmap")
        corr_matrix = coal[columns].corr()
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f', ax=ax)
        st.pyplot(fig)

        # Stationarity test using ADF
        st.subheader("Stationarity Test Using ADF")
        for column in columns:
            result = adfuller(coal[column].dropna())
            st.write(f"ADF Statistic for {column}: {result[0]}")
            st.write(f"p-value: {result[1]}")
            st.write("Critical values:")
            for key, value in result[4].items():
                st.write(f"\t{key}: {value:.3f}")

        # Forecasting
        st.subheader("Coal Price Forecasting with GradientBoostingRegressor")

        # Select target column for forecasting
        target_column = st.selectbox("Select the target column for forecasting", columns)

        # Select start date and end date
        start_date = st.date_input("Select the start date", pd.to_datetime('2024-07-01'))
        end_date = st.date_input("Select the end date")

        # Select confidence interval
        confidence_interval = st.selectbox("Select confidence interval", [95, 96, 97, 98, 99, 100])

        if st.button("Forecast"):
            train_start = pd.Timestamp('2020-02-04')
            train_end = pd.Timestamp('2023-12-29')
            test_start = pd.Timestamp('2024-01-01')
            test_end = pd.Timestamp(end_date)
            X_train, X_test, y_train, y_test, scaler = preprocess_data(coal, target_column, train_start, train_end, test_start, test_end)
            
            # Check for NaNs or infinite values
            st.write("Checking for NaN or infinite values in X_train, X_test, y_train, y_test:")
            st.write(pd.DataFrame(X_train).isna().sum())
            st.write(pd.DataFrame(X_test).isna().sum())
            st.write(pd.Series(y_train).isna().sum())
            st.write(pd.Series(y_test).isna().sum())
            
            model = GradientBoostingRegressor(n_estimators=100, random_state=42)
            model.fit(X_train, y_train)

            # Generate forecasts
            y_pred_train = model.predict(X_train)
            y_pred_test = model.predict(X_test)

            # Calculate MAPE
            train_mape = mean_absolute_percentage_error(y_train, y_pred_train)
            test_mape = mean_absolute_percentage_error(y_test, y_pred_test)
            st.write(f"Train MAPE: {train_mape:.2f}%")
            st.write(f"Test MAPE: {test_mape:.2f}%")

            # Plot actual vs forecasted values
            st.write("Actual vs Forecasted Values")

            # Create a DataFrame for actual vs predicted values
            results_df = pd.DataFrame({
                'Date': pd.date_range(start=test_start, periods=len(y_test), freq='D'),
                'Actual': y_test.values,
                'Forecasted': y_pred_test
            })

            # Plotting
            fig, ax = plt.subplots(figsize=(12, 6))
            ax.plot(results_df['Date'], results_df['Actual'], label="Actual")
            ax.plot(results_df['Date'], results_df['Forecasted'], label="Forecasted", linestyle='--')
            ax.set_xlabel("Date")
            ax.set_ylabel("Values")
            ax.set_title("Actual vs Forecasted Values")
            ax.legend()
            st.pyplot(fig)

            # Display results as a table
            st.write("Results Table")
            st.write(results_df[['Date', 'Actual', 'Forecasted']])

            # Calculate success rate
            success_rate = (1 - test_mape) * 100
            st.write(f"Success Rate of Forecasting: {success_rate:.2f}%")

            # Forecast for the next period
            st.subheader("Forecast from July 2024")

            # Generate future dates starting from July 2024
            future_dates = pd.date_range(start=pd.Timestamp('2024-07-01'), periods=(pd.Timestamp(end_date) - pd.Timestamp('2024-07-01')).days + 1)

            last_known_data = X_test[-1].reshape(1, -1)  # Use the last data point from the test set
            forecasted_values = []
            for i in range(len(future_dates)):
                next_pred = model.predict(last_known_data)
                forecasted_values.append(next_pred[0])
                last_known_data = np.append(last_known_data[:, 1:], next_pred).reshape(1, -1)  # Slide the window

            forecast_df = pd.DataFrame({
                'Date': future_dates,
                'Forecasted': forecasted_values
            })

            # Plotting the forecast
            fig, ax = plt.subplots(figsize=(12, 6))
            ax.plot(results_df['Date'], results_df['Actual'], label="Actual")
            ax.plot(results_df['Date'], results_df['Forecasted'], label="Forecasted", linestyle='--')
            ax.plot(forecast_df['Date'], forecast_df['Forecasted'], label="Future Forecasted", linestyle='-.', color='red')
            ax.set_xlabel("Date")
            ax.set_ylabel("Values")
            ax.set_title(f"Forecast from July 2024 with Confidence Interval {confidence_interval}%")
            ax.legend()
            st.pyplot(fig)

            # Display forecast results as a table
            st.write("Forecast Results Table")
            st.write(forecast_df[['Date', 'Forecasted']])

if __name__=='__main__':
    main()
