import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pandas_datareader as data
import yfinance as yf
import tensorflow as tf
from datetime import date
from keras.models import load_model
import streamlit as st
import streamlit as st
from datetime import date
import yfinance as yf
from prophet import Prophet
from prophet.plot import plot_plotly
from plotly import graph_objs as go
start = '2010-01-01'
end =  date.today().strftime("%Y-%m-%d")
st.title('Stock Future Predicter')
use_input = st.text_input('Enter stock Ticker', 'AAPL')  ##############
n_years = st.slider('Years of prediction:', 1, 4)
period = n_years * 365
if st.button('Predict'):
    df = yf.download(use_input, start, end)
    # describing data
    st.subheader('Data From 2010-2023')
    st.write(df.describe())
    # maps
    st.subheader('closing Price VS Time Chart ')
    fig = plt.figure(figsize=(10, 5))
    plt.plot(df.Close, color='yellow')
    plt.legend()
    st.pyplot(fig)
    st.subheader('closing Price VS Time Chart with 100 moving Average  ')
    ma100 = df.Close.rolling(100).mean()
    fig = plt.figure(figsize=(10, 5))
    plt.plot(ma100, color='red')
    plt.plot(df.Close, color='yellow')
    plt.legend()
    st.pyplot(fig)
    st.subheader('closing Price VS Time Chart with 100 & 200 moving Average  ')
    ma100 = df.Close.rolling(100).mean()
    ma200 = df.Close.rolling(200).mean()
    fig = plt.figure(figsize=(10, 5))
    plt.plot(ma100, color='red')
    plt.plot(ma200, color='green')
    plt.plot(df.Close, color='yellow')
    plt.legend()
    st.pyplot(fig)
    # spltting data into train test
    data_training = pd.DataFrame(df['Close'][0:int(len(df) * 0.70)])
    data_testing = pd.DataFrame(df['Close'][int(len(df) * 0.70):int(len(df))])
    print(' taining ', data_training.shape)
    print(' testing ', data_testing.shape)
    from sklearn.preprocessing import MinMaxScaler
    scaler = MinMaxScaler(feature_range=(0, 1))
    data_training_array = scaler.fit_transform(data_training)
    # load Model
    model = load_model('model.h5')
    # testing past
    pass_100_days = data_training.tail(100)
    final_df = pd.concat([pass_100_days, data_testing], ignore_index=True)
    input_data = scaler.fit_transform(final_df)
    x_test = []
    y_test = []
    for i in range(100, input_data.shape[0]):
        x_test.append(input_data[i - 100:i])
        y_test.append(input_data[i, 0])
    x_test, y_test = np.array(x_test), np.array(y_test)
    y_predicted = model.predict(x_test)
    scaler = scaler.scale_
    scale_factor = 1 / scaler[0]
    y_predicted = y_predicted * scale_factor
    y_test = y_test * scale_factor


    @st.cache_data
    def load_data(ticker):
        data = yf.download(ticker, start, end)
        data.reset_index(inplace=True)
        return data
    data_load_state = st.text('Loading data...')
    data = load_data(use_input)
    data_load_state.text('Loading data... done!')
    # Predict forecast with Prophet.
    df_train = data[['Date', 'Close']]
    df_train = df_train.rename(columns={"Date": "ds", "Close": "y"})

    m = Prophet()
    m.fit(df_train)
    future = m.make_future_dataframe(periods=period)
    forecast = m.predict(future)

    # Show and plot forecast

    st.write(f'Forecast plot for {n_years} years')
    fig1 = plot_plotly(m, forecast)
    st.plotly_chart(fig1)

    st.write("Forecast components")
    fig2 = m.plot_components(forecast)
    st.write(fig2)


    # final graph
    def plot_transparent_graph():
        st.subheader('prediction vs Original')
        fig2 = plt.figure(figsize=(12, 6))
        plt.plot(y_test, 'b', label='Original Price')
        plt.plot(y_predicted, 'r', label='prdicted Price')
        plt.style.use('dark_background')
        plt.xlabel('time')
        plt.ylabel('price')
        plt.legend()
        st.pyplot(fig2)


    def main():
        st.title('Stock Price Predicted Analysis')

        # Call the function to plot the transparent graph
        plot_transparent_graph()

        # Other interactive elements and text can be added here as needed
        # ...
    if __name__ == "__main__":
        main()
        