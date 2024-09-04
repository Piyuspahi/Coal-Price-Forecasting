
'''
# **Problem Statement**

#  Coal price forecasting

#Coal price are obtained as need be rather than forecasting on 
   what might be the coal price.
   This sometime lead to spending more cost procuring the coal,
   which is the row material for steel manufacturing.
 
# Business Objective 
   Maximize cost saving and profitability by optimizing procurement and sourcing
   strategies through acurate price forecasting.
   
# Business constraint
   Minimise impact of price volatility on production cost. 
  

# **Success Criteria**

# Business success criteria
   Achiveing a 10% in increase in profit margins through optimized procurement 
   and pricing strategies.
  
# Machinelearning success criteria
   Achive an accuracy of aleast 95%.
   
# Economic success criteria
   Generating a 20% increase in revenue from coal and iron ore sales within 
   the first year of implementation.
'''  
   
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as snb
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
import joblib
import pickle
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from feature_engine.outliers import Winsorizer
from sqlalchemy import create_engine, text
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error
from math import sqrt
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from sklearn.metrics import mean_absolute_percentage_error

#import data

data = pd.read_csv(r"C:\Users\Piyus Pahi\Documents\M.L Project\Data set - EDA.csv")
data

        
# First 5 rows   
print(data.head())

# Last 5 rows
data.tail()

# Describe the data
data.describe()

# Information about the data
data.info()


#MySql database connection 

user = "root" #User
pw = "965877" #password
db = "coal_db" #database
engine = create_engine(f"mysql+pymysql://{user}:{pw}@localhost/{db}")

data.to_sql("coal", con = engine, if_exists = "replace", index = False)

sql = "select * from coal;"
coal = pd.read_sql_query(sql, engine.connect())
coal.columns

        
# First 5 rows   
print(coal.head())

# Last 5 rows
coal.tail()

# Describe the data
coal.describe()

# Information about the data
coal.info()

# Sum of null value
coal.isnull().sum()


# ****EDA****

# List of column names to iterate over
columns = [
    'Coal_RB_4800_FOB_London_Close_USD',
    'Coal_RB_5500_FOB_London_Close_USD',
    'Coal_RB_5700_FOB_London_Close_USD',
    'Coal_RB_6000_FOB_CurrentWeek_Avg_USD',
    'Coal_India_5500_CFR_London_Close_USD',
    'Price_WTI',
    'Price_Brent_Oil',
    'Price_Dubai_Brent_Oil',
    'Price_ExxonMobil',
    'Price_Shenhua',
    'Price_All_Share',
    'Price_Mining',
    'Price_LNG_Japan_Korea_Marker_PLATTS',
    'Price_ZAR_USD',
    'Price_Natural_Gas',
    'Price_ICE',
    'Price_Dutch_TTF',
    'Price_Indian_en_exg_rate'
]

# First moment business decission (mean, median, mode)

for column in columns:
    mean_value = coal[column].mean()
    median_value = coal[column].median()
    mode_value = coal[column].mode()[0]  # mode() returns a Series; [0] gets the first mode

    print(f"{column}:")
    print(f"  Mean: {mean_value}")
    print(f"  Median: {median_value}")
    print(f"  Mode: {mode_value}\n")

# 2nd moment business decission(standard deviation, variance, range)

for column in columns:
    variance = coal[column].var()
    std_dev = coal[column].std()
    value_range = coal[column].max() - coal[column].min()

    print(f"{column}:")
    print(f"  Variance: {variance}")
    print(f"  Standard Deviation: {std_dev}")
    print(f"  Range: {value_range}\n")


# 3rd moment business decission

for column in columns:
    skewness = coal[column].skew()
   

    print(f"{column}:")
    print(f"  Skewness: {skewness}")


# 4th moment business decission

for column in columns:
    kurtosis = coal[column].kurt()

    print(f"{column}:")
    print(f"  Kurtosis: {kurtosis}\n") 
    
    
#Create scatter plots for pairs of columns

for i in range(len(columns)):
    for j in range(i + 1, len(columns)):
        plt.figure(figsize=(8, 6))
        snb.scatterplot(x=coal[columns[i]], y=coal[columns[j]])
        plt.xlabel(columns[i])
        plt.ylabel(columns[j])
        plt.title(f'Scatter Plot of {columns[i]} vs {columns[j]}')
        plt.show()    

# plot graphs univariate    

"pair plot"
snb.pairplot(coal)
plt.show()


"""Density plot"""

for column in columns:
    plt.figure(figsize=(8, 6))
    snb.kdeplot(data=coal[column], shade=True)
    plt.xlabel(column)
    plt.ylabel('Density')
    plt.title(f'Density Plot of {column}')
    plt.show()

"""Histogram"""

for column in columns:
    plt.hist(coal[column], bins=30)  # Adjust the number of bins as needed
    plt.xlabel(column)
    plt.ylabel('Frequency')
    plt.title(f'{column} - Histogram')
    plt.show()

"""Bar graph"""

for column in columns:
    # Bar chart
    x = np.arange(1, len(coal[column]) + 1)
    plt.bar(x, coal[column])
    plt.xlabel('Index')
    plt.ylabel(column)
    plt.title(f'{column} - Bar Chart')
    plt.show()
  
    
"""Data preprocessing"""

#convert date column in to datetime format
coal['Date'] = pd.to_datetime(coal['Date'],format="%d-%m-%Y", dayfirst=True, errors='coerce') 
coal.sort_values(by=['Date'], inplace=True, ascending=True)

coal.head()

#set index to the date column
#coal.set_index('Date')

# Ensure the index is a DatetimeIndex
#coal.index = pd.to_datetime(coal.index)
#coal.head()

# Check the index type to confirm
#print(type(coal.index))

#find the missing values
coal.isna().sum()

#plot graphically to represent missing values

for column in columns:
    plt.figure(figsize=(20, 10), dpi=80)
    plt.plot(coal['Date'], coal[column], linewidth=2)
    plt.xlabel('Date')
    plt.ylabel(column)
    plt.title(f'Time Series Plot of {column}')
    plt.grid(True)
    plt.show()


#each column apply forward fill method

for column in columns:
    coal[column] = coal[column].fillna(method='ffill')
    
    
#checking the null values
coal.isnull().sum()

#apply backward interpolation for the rest columns where the missing values can not be removed by forward fill
coal.Price_Indian_en_exg_rate = coal.Price_Indian_en_exg_rate.fillna(method = 'bfill')

coal.isnull().sum()

#check the outliers by using boxplot

coal.plot(kind = 'box', subplots = True, sharey = False, figsize = (60, 25)) 
# Increase spacing between subplots
plt.subplots_adjust(wspace = 0.75) # ws is the width of the padding between subplots, as a fraction of the average Axes width.
plt.show()

# #### Outlier analysis: 
winsor = Winsorizer(capping_method = 'iqr', # choose  IQR rule boundaries or gaussian for mean and std
                          tail = 'both', # cap left, right or both tails 
                          fold = 1.5,
                          variables = ['Coal_RB_4800_FOB_London_Close_USD','Coal_RB_5500_FOB_London_Close_USD','Coal_RB_5700_FOB_London_Close_USD',
                                       'Coal_RB_6000_FOB_CurrentWeek_Avg_USD','Coal_India_5500_CFR_London_Close_USD','Price_WTI',
                                       'Price_Brent_Oil','Price_Dubai_Brent_Oil','Price_ExxonMobil','Price_Shenhua','Price_All_Share',
                                       'Price_Mining','Price_LNG_Japan_Korea_Marker_PLATTS','Price_ZAR_USD','Price_Natural_Gas','Price_ICE',
                                       'Price_Dutch_TTF','Price_Indian_en_exg_rate'])



outlier = winsor.fit(coal[['Coal_RB_4800_FOB_London_Close_USD','Coal_RB_5500_FOB_London_Close_USD','Coal_RB_5700_FOB_London_Close_USD',
             'Coal_RB_6000_FOB_CurrentWeek_Avg_USD','Coal_India_5500_CFR_London_Close_USD','Price_WTI',
             'Price_Brent_Oil','Price_Dubai_Brent_Oil','Price_ExxonMobil','Price_Shenhua','Price_All_Share',
             'Price_Mining','Price_LNG_Japan_Korea_Marker_PLATTS','Price_ZAR_USD','Price_Natural_Gas','Price_ICE',
             'Price_Dutch_TTF','Price_Indian_en_exg_rate']])



coal[['Coal_RB_4800_FOB_London_Close_USD','Coal_RB_5500_FOB_London_Close_USD','Coal_RB_5700_FOB_London_Close_USD',
             'Coal_RB_6000_FOB_CurrentWeek_Avg_USD','Coal_India_5500_CFR_London_Close_USD','Price_WTI',
             'Price_Brent_Oil','Price_Dubai_Brent_Oil','Price_ExxonMobil','Price_Shenhua','Price_All_Share',
             'Price_Mining','Price_LNG_Japan_Korea_Marker_PLATTS','Price_ZAR_USD','Price_Natural_Gas','Price_ICE',
             'Price_Dutch_TTF','Price_Indian_en_exg_rate']] = outlier.transform(coal[['Coal_RB_4800_FOB_London_Close_USD','Coal_RB_5500_FOB_London_Close_USD','Coal_RB_5700_FOB_London_Close_USD',
                          'Coal_RB_6000_FOB_CurrentWeek_Avg_USD','Coal_India_5500_CFR_London_Close_USD','Price_WTI',
                          'Price_Brent_Oil','Price_Dubai_Brent_Oil','Price_ExxonMobil','Price_Shenhua','Price_All_Share',
                          'Price_Mining','Price_LNG_Japan_Korea_Marker_PLATTS','Price_ZAR_USD','Price_Natural_Gas','Price_ICE',
                          'Price_Dutch_TTF','Price_Indian_en_exg_rate']])
                                                                                      
#again checking the outliers
coal.plot(kind = "box",subplots = True,sharey = False,figsize =(70,40))
plt.subplots_adjust(wspace = 1.5) # ws is the width of the padding between subplots, as a fraction of the average Axes width.
plt.show()
                                                                                    

# Save the winsorizer model 
joblib.dump(outlier, 'winsor')

coal.isnull().sum()
                                                                                     
# replace capped values with NaN
for col in ['Coal_RB_4800_FOB_London_Close_USD','Coal_RB_5500_FOB_London_Close_USD','Coal_RB_5700_FOB_London_Close_USD',
             'Coal_RB_6000_FOB_CurrentWeek_Avg_USD','Coal_India_5500_CFR_London_Close_USD','Price_WTI',
             'Price_Brent_Oil','Price_Dubai_Brent_Oil','Price_ExxonMobil','Price_Shenhua','Price_All_Share',
             'Price_Mining','Price_LNG_Japan_Korea_Marker_PLATTS','Price_ZAR_USD','Price_Natural_Gas','Price_ICE',
             'Price_Dutch_TTF','Price_Indian_en_exg_rate']:  # iterate over numeric columns
    coal[col] = np.where((coal[col] == coal[col].max()) | (coal[col] == coal[col].min()), np.nan, coal[col])

print(coal)


#applying interpolation based on time
#each column apply forward fill

for column in columns:
    coal[column] = coal[column].fillna(method='ffill')
    
#checking the null values
coal.isnull().sum()

#apply backward interpolation for the rest columns where the missing values can not be removed by forward fill
for column in columns:
    coal[column] = coal[column].fillna(method='bfill')
    
#checking the null values    
coal.isnull().sum()    


"""stationary test"""

'''Stationarity means that the statistical properties of a time series i.e. mean, variance and covariance do not change over time'''

'''############## Augmented Dickey Fuller (“ADF”) test'''

#By applying ADF techniques to prove know that the data is stationary or not
#determine null and alternate hypothesis

'''# null hypothesis Ho = Time series is non-stationary in nature
     alternate hypothesis Ha = Time series is stationary in nature'''

#if ADF statistic < Critical value ,then reject the null hypothesis
## if ADF statistic > critical value, then failed to reject the null hypothesis

#checking data is stationary or not by using graph

for column in columns:
    plt.plot(coal[column])
    plt.title(f"Plot of {column}")
    plt.show()
    
    x = coal[column].values
    result = adfuller(x)
    print(f"ADF Statistic for {column}: {result[0]}")
    print(f"p-value: {result[1]}")
    print("Critical values:")
    for key, value in result[4].items():
        print(f"\t{key}: {value:.3f}")
    
    if result[0] < result[4]["5%"]:
        print(f"Reject Ho - {column} is Stationary")
    else:
        print(f"Failed to Reject Ho - {column} is Non-stationary")
    
    print("\n")


#Here the insight is only Price_WTI column is stationary and others are non-stationary.

#calculating correlation
# Drop the date column if it's not needed for correlation analysis
coal_numeric = coal.drop(columns=['Date'])
coal_numeric

# Calculate the correlation matrix
corr_matrix = coal_numeric.corr()
corr_matrix

#show in heatmap
plt.figure(figsize=(10, 8))
snb.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Correlation Heatmap')
plt.show()


#perform Auto_EDA

# sweetviz
##########
import sweetviz
my_report_s = sweetviz.analyze([coal,"coal"])
my_report_s.show_html("report.html")

#autoviz
########
from autoviz.AutoViz_Class import AutoViz_Class
av = AutoViz_Class()
a = av.AutoViz(r"C:\Users\Piyus Pahi\Documents\M.L Project\Data set - EDA.csv",chart_format='html')
import os
os.getcwd()

# D-Tale
########
import dtale
my_report_d = dtale.show(coal)
my_report_d.open_browser()
import os
os.getcwd()

# Save the DataFrame to a CSV file
coal.to_csv('pre_processed_data.csv', index=False)
import os
os.getcwd()

'''# #####################################  Forecast Model Building ##############################'''


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_percentage_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

# Convert the 'Date' column to datetime if not already done
coal['Date'] = pd.to_datetime(coal['Date'], dayfirst=True, errors='coerce')

# Define the date range for training and testing
train_start = pd.Timestamp('2020-02-04')
train_end = pd.Timestamp('2023-12-29')
test_start = pd.Timestamp('2024-01-01')
test_end = pd.Timestamp('2024-06-28')

# Define the columns for forecasting
target_columns = ['Price_WTI', 'Price_Brent_Oil', 'Price_Dubai_Brent_Oil', 'Price_ExxonMobil', 'Price_Shenhua', 
                  'Price_All_Share', 'Price_Mining', 'Price_LNG_Japan_Korea_Marker_PLATTS', 'Price_ZAR_USD', 
                  'Price_Natural_Gas', 'Price_ICE', 'Price_Dutch_TTF', 'Price_Indian_en_exg_rate']

# Function to create sequences for LSTM
def create_sequences(data, n_steps):
    X, y = [], []
    for i in range(len(data) - n_steps):
        X.append(data[i:i + n_steps])
        y.append(data[i + n_steps])
    return np.array(X), np.array(y)

# Parameters
n_steps = 12  # Number of time steps for sequences
epochs = 50
batch_size = 32

# Loop through each target column
for target_column in target_columns:
    print(f"Forecasting for {target_column}")

    if target_column not in coal.columns:
        print(f"Column {target_column} not found in the dataset.")
        continue

    # Extract data
    target_data = coal[['Date', target_column]].dropna()
    
    # Split data into training and testing sets using boolean indexing
    train_data = target_data[(target_data['Date'] >= train_start) & (target_data['Date'] <= train_end)][target_column]
    test_data = target_data[(target_data['Date'] >= test_start) & (target_data['Date'] <= test_end)][target_column]

    if train_data.empty or test_data.empty:
        print(f"No data available for {target_column} in the specified date range.")
        continue

    # Scale data
    scaler = MinMaxScaler()
    train_scaled = scaler.fit_transform(train_data.values.reshape(-1, 1))
    test_scaled = scaler.transform(test_data.values.reshape(-1, 1))

    # Create sequences
    X_train, y_train = create_sequences(train_scaled, n_steps)
    X_test, y_test = create_sequences(test_scaled, n_steps)

    # Reshape for LSTM input
    X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
    X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))

    # Define LSTM model
    model = Sequential([
        LSTM(50, activation='relu', input_shape=(n_steps, 1)),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse')

    # Fit model
    model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, verbose=0)

    # Make predictions on training data
    y_train_pred = model.predict(X_train, verbose=0)
    y_train_pred = scaler.inverse_transform(y_train_pred)

    # Make predictions on testing data
    y_test_pred = model.predict(X_test, verbose=0)
    y_test_pred = scaler.inverse_transform(y_test_pred)

    # Calculate MAPE for training data
    y_train_inverse = scaler.inverse_transform(y_train.reshape(-1, 1))
    train_mape = mean_absolute_percentage_error(y_train_inverse, y_train_pred)
    print(f'MAPE for {target_column} (Train): {train_mape:.2f}')

    # Calculate MAPE for testing data
    y_test_inverse = scaler.inverse_transform(y_test.reshape(-1, 1))
    test_mape = mean_absolute_percentage_error(y_test_inverse, y_test_pred)
    print(f'MAPE for {target_column} (Test): {test_mape:.2f}')

    # Plot results
    plt.figure(figsize=(10, 5))
    plt.plot(target_data['Date'], target_data[target_column], label='Actual')
    plt.plot(target_data['Date'][(target_data['Date'] >= train_start) & (target_data['Date'] <= train_end)][n_steps:], 
             y_train_pred, label='Train Forecast', color='orange')
    plt.plot(target_data['Date'][(target_data['Date'] >= test_start) & (target_data['Date'] <= test_end)][n_steps:], 
             y_test_pred, label='Test Forecast', color='green')
    plt.legend()
    plt.title(f'Forecast vs Actuals for {target_column}')
    plt.show()

'''################### Regression Model Building ##################'''

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, ExtraTreesRegressor
from sklearn.ensemble import StackingRegressor
from xgboost import XGBRegressor
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM
from sklearn.preprocessing import MinMaxScaler
import joblib
import tensorflow as tf

# Custom MAPE loss function for LSTM
def mape_loss(y_true, y_pred):
    return tf.reduce_mean(tf.abs((y_true - y_pred) / tf.clip_by_value(tf.abs(y_true), 1e-8, tf.reduce_max(tf.abs(y_true))))) * 100

# Specify column names
variable_factors = ['Price_WTI', 'Price_Brent_Oil', 'Price_Dubai_Brent_Oil',
                    'Price_ExxonMobil', 'Price_Shenhua', 'Price_All_Share', 
                    'Price_Mining', 'Price_LNG_Japan_Korea_Marker_PLATTS',
                    'Price_ZAR_USD', 'Price_Natural_Gas', 'Price_ICE',
                    'Price_Dutch_TTF','Price_Indian_en_exg_rate']

coal_prices_cols = ['Coal_RB_4800_FOB_London_Close_USD', 'Coal_RB_5500_FOB_London_Close_USD',
                    'Coal_RB_5700_FOB_London_Close_USD', 'Coal_RB_6000_FOB_CurrentWeek_Avg_USD', 
                    'Coal_India_5500_CFR_London_Close_USD']

# Load and preprocess your data
# Assume coal DataFrame is already defined and sorted by 'Date'
# coal = pd.read_csv('your_coal_data.csv')  # Example: Load your data
# coal['Date'] = pd.to_datetime(coal['Date'])

# Sort the data by date to ensure chronological order
coal = coal.sort_values(by='Date')  # Ensure chronological order

# Split the data into training and test sets based on date
train_data = coal[(coal['Date'] >= '2020-04-02') & (coal['Date'] <= '2023-12-29')]
test_data = coal[(coal['Date'] >= '2024-01-01') & (coal['Date'] <= '2024-06-28')]

# Extract input and output data
X_train = train_data[variable_factors].values
Y_train = train_data[coal_prices_cols].values
X_test = test_data[variable_factors].values
Y_test = test_data[coal_prices_cols].values

# Scale the inputs
scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Save the scaler
joblib.dump(scaler, 'minmax_scaler.joblib')

# Reshape data for LSTM input [samples, time steps, features]
X_train_lstm = X_train_scaled.reshape((X_train_scaled.shape[0], 1, X_train_scaled.shape[1]))
X_test_lstm = X_test_scaled.reshape((X_test_scaled.shape[0], 1, X_test_scaled.shape[1]))

# Define LSTM model with custom MAPE loss function
def build_lstm(input_shape):
    model = Sequential()
    model.add(LSTM(units=50, activation='relu', input_shape=input_shape))
    model.add(Dense(len(coal_prices_cols)))  # Output layer size matches coal price columns
    model.compile(optimizer='adam', loss=mape_loss)  # Use custom MAPE loss
    return model

# Train LSTM model
lstm_model = build_lstm((X_train_lstm.shape[1], X_train_lstm.shape[2]))
lstm_model.fit(X_train_lstm, Y_train, epochs=100, batch_size=32, verbose=2)

# Save LSTM model
lstm_model.save('lstm_model.h5')

# Forecast the variable factors
forecast_train = lstm_model.predict(X_train_lstm)
forecast_test = lstm_model.predict(X_test_lstm)

# Function to train and evaluate models using MAPE for each coal price column
def train_and_evaluate_each_target(models, forecast_train, forecast_test, Y_train, Y_test, target_names):
    results = {}
    overall_best_model_name = None
    overall_best_model = None
    overall_best_mape = float('inf')

    # Iterate over each target column
    for idx, target_name in enumerate(target_names):
        print(f'\nEvaluating models for target: {target_name}')
        results[target_name] = {}

        # Get the training and testing data for this specific target
        y_train_target = Y_train[:, idx]
        y_test_target = Y_test[:, idx]

        # Train and evaluate each model
        for name, model in models:
            model.fit(forecast_train, y_train_target)
            train_predictions = model.predict(forecast_train)
            test_predictions = model.predict(forecast_test)
            
            # Calculate MAPE for both train and test data
            train_mape = mean_absolute_percentage_error(y_train_target, train_predictions)
            test_mape = mean_absolute_percentage_error(y_test_target, test_predictions)
            
            results[target_name][name] = {'train_mape': train_mape, 'test_mape': test_mape}
            print(f'{name} Train MAPE for {target_name}: {train_mape:.2f}')
            print(f'{name} Test MAPE for {target_name}: {test_mape:.2f}')

            # Check for the best model across all targets based on average test MAPE
            if test_mape < overall_best_mape:
                overall_best_mape = test_mape
                overall_best_model_name = name
                overall_best_model = model

    return results, (overall_best_model_name, overall_best_model, overall_best_mape)

# Define models
models = [
    ('XGBoost', XGBRegressor()),
    ('GradientBoosting', GradientBoostingRegressor()),
    ('RandomForest', RandomForestRegressor()),
    ('ExtraTrees', ExtraTreesRegressor())
]

# Evaluate models
results, (best_model_name, best_model, best_mape) = train_and_evaluate_each_target(models, forecast_train, forecast_test, Y_train, Y_test, coal_prices_cols)

# Train and evaluate stacking model
stacking_model = StackingRegressor(
    estimators=models,
    final_estimator=GradientBoostingRegressor()
)

# Calculate and print MAPE for the stacking model for each coal price target
for idx, target_name in enumerate(coal_prices_cols):
    stacking_model.fit(forecast_train, Y_train[:, idx])
    stacking_train_predictions = stacking_model.predict(forecast_train)
    stacking_test_predictions = stacking_model.predict(forecast_test)
    
    stacking_train_mape = mean_absolute_percentage_error(Y_train[:, idx], stacking_train_predictions)
    stacking_test_mape = mean_absolute_percentage_error(Y_test[:, idx], stacking_test_predictions)
    
    print(f'Stacking Model Train MAPE for {target_name}: {stacking_train_mape:.2f}')
    print(f'Stacking Model Test MAPE for {target_name}: {stacking_test_mape:.2f}')
    
    # Check if stacking model is the best overall
    if stacking_test_mape < best_mape:
        best_mape = stacking_test_mape
        best_model_name = 'Stacking Model'
        best_model = stacking_model

# Save the best overall model
joblib.dump(best_model, f'best_model_{best_model_name}.joblib')
print(f'\nBest overall model is {best_model_name} with Test MAPE: {best_mape:.2f}')

# Combine train and test data for future use
combined_data = pd.concat([train_data, test_data], axis=0)
combined_data.to_csv('combined_coal_data.csv', index=False)  # Save combined data to a CSV file
print("\nCombined train and test data saved to 'combined_coal_data.csv'.")

import os
os.getcwd()






































































































































































