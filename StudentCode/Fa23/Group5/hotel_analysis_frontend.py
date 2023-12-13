import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX
from matplotlib import pyplot as plt
import streamlit as st
from sklearn.metrics import mean_squared_error
from math import sqrt

best_params = [(1,1,2), (2,1,2,12)]

def clean_data_single_location(df, location):
    # strip leading and trailing whitespace from location names in df
    df['Location Name'] = df['Location Name'].str.strip()
    df = df[df['Location Name'] == location]
    df = df.drop(['Location Name'], axis=1)
    
    prev_size = df.shape[0]
    df = df[df["Filer Type"] == 50]
    print(f"Num quarterly records: {prev_size - df.shape[0]}")
    print(f"Num monthly records: {df.shape[0]}")

    # drop filer type
    df = df.drop(['Filer Type'], axis=1)

    # create a datetime column from obligation end date
    df["Obligation End Date (YYYYMMDD)"] = pd.to_datetime(df["Obligation End Date (YYYYMMDD)"], format="%Y%m%d")

    # set index to datetime
    df = df.set_index(["Obligation End Date (YYYYMMDD)"])

    # get range of dates
    indices = pd.date_range(start=df.index.min(), end=df.index.max(), freq='M')
    df.index = pd.DatetimeIndex(df.index).to_period('M')

    y = df["Total Room Receipts"]
    x = df["Unit Capacity"]

    return x, y, indices

def train_and_test(x, y, xtest, ytest, params):
    pdq, seasonal_pdq = params
    # Fit the model
    model = SARIMAX(y, exog=x, order=pdq, seasonal_order=seasonal_pdq, enforce_stationarity=False, enforce_invertibility=False)
    model_fit = model.fit(disp=False)

    # make predictions
    yhat = model_fit.predict(start=len(y), end=len(y)+len(ytest)-1, exog=xtest)
    
    # return aic
    return model_fit.aic, yhat, model_fit

def train_predict_and_plot(df_train, df_test, location):
    x, y, _ = clean_data_single_location(df_train, location)
    xtest, ytest, indices = clean_data_single_location(df_test, location)

    aic, yhat, model_fit = train_and_test(x, y, xtest, ytest, best_params)

    # test model with start as first date in test data and end as last date in test data before 2019
    print(f"Test data contains {ytest.shape[0]} months with start date {indices[0]} and end date {indices[-1]}")

    # last item in indices in the year 2019
    for i in range(0, ytest.shape[0]):
        if indices[i].year == 2019:
            end_idx = i
            end = indices[i]

    print(f"Updating end date to {end} at index {end_idx}")
    yhat = model_fit.get_prediction(start=len(y), end=len(y)+end_idx, exog=xtest[y.index[-1]:end], dynamic=indices[0])

    # RMSE
    rmse = sqrt(mean_squared_error(ytest[:end_idx+1], yhat.predicted_mean))
    print(f"RMSE: {rmse}")

    # plot predictions and expected results
    fig = plt.figure(figsize=(8, 6))
    plt.plot(indices[:end_idx+1], yhat.predicted_mean, label='Predicted')
    plt.plot(indices[:end_idx+1], ytest[:end_idx+1], label='Expected')
    plt.ylabel("Total Room Receipts")
    plt.xlabel("Date")
    plt.xticks(rotation=45)
    plt.title("SARIMA Predictions vs Expected Results")
    plt.legend()

    return fig, rmse, indices[:end_idx+1]

def get_date_range(df, location):
    df = df[df["Location Name"] == location]
    df = df.drop(['Location Name'], axis=1)
    df = df.drop(['Filer Type'], axis=1)
    df["Obligation End Date (YYYYMMDD)"] = pd.to_datetime(df["Obligation End Date (YYYYMMDD)"], format="%Y%m%d")
    df = df.set_index(["Obligation End Date (YYYYMMDD)"])
    return df.index.min().strftime("%Y-%m-%d"), df.index.max().strftime("%Y-%m-%d")

df_train = pd.read_csv("./Om Data Loan/TrainingLoanData.csv")
df_test = pd.read_csv("./Om Data Loan/TestingLoanData.csv")
df_train = df_train[df_train["Filer Type"] == 50]
df_test = df_test[df_test["Filer Type"] == 50]
locations = list(set(df_train["Location Name"].unique()).intersection(set(df_test["Location Name"].unique())))
location = st.selectbox("Select a location", locations)

button = st.button("Run Analysis")
if button:
    st.title("Hotel Analysis")
    st.write(f"Location name: {location}")
    st.write(f"Location city: {df_train[df_train['Location Name'] == location]['Location City'].iloc[0]}")
    st.write(f"Using data from {get_date_range(df_train, location)[0]} to {get_date_range(df_train, location)[1]} for training")
    fig, rmse, indices = train_predict_and_plot(df_train, df_test, location.strip())
    st.write(f"Generating predictions from {indices[0].strftime('%Y-%m-%d')} to {indices[-1].strftime('%Y-%m-%d')}")
    st.write(f"RMSE: {rmse}")
    st.plotly_chart(fig, sharing='streamlit', config={'displayModeBar': False}, use_container_width=True)