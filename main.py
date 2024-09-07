import plotly as plotly
import streamlit
import streamlit as anu
import pandas as pd, numpy as np, yfinance as yf
import plotly.express as px
from prophet import Prophet
from prophet.plot import plot_plotly

#-----------------------------------------------------------new code------------
import pyrebase
import streamlit as st
from datetime import datetime

#configuration
firebaseConfig = {
  'apiKey': "AIzaSyC46QOlqNdRDWXYNeJNjoGOugl-dptAGkE",
  'authDomain': "test-streamlit-demo.firebaseapp.com",
  'projectId': "test-streamlit-demo",
  'storageBucket': "test-streamlit-demo.appspot.com",
  'databaseURL':" https://test-streamlit-demo-default-rtdb.europe-west1.firebasedatabase.app/",
  'messagingSenderId': "690728706107",
  'appId': "1:690728706107:web:dceaf0eb27849c0c1fbc1e",
  'measurementId': "G-B3PR2Z8GCB"
};
# Firebase Authentication
firebase = pyrebase.initialize_app(firebaseConfig)
auth = firebase.auth()

# Database
db = firebase.database()
storage = firebase.storage()
st.sidebar.title("Our community  stock app")

# Authentication
choice = st.sidebar.selectbox('login/Signup', ['Login', 'Sign up'])

# Obtain User Input for email and password
email = st.sidebar.text_input('Please enter your email address')
password = st.sidebar.text_input('Please enter your password',type = 'password')

#App

# Sign up Block
if choice == 'Sign up':
  handle = st.sidebar.text_input(
    'Please input your app handle name', value='Default')
  submit = st.sidebar.button('Create my account')

  if submit:
    user = auth.create_user_with_email_and_password(email, password)
    st.success('Your account is created suceesfully!')
    st.balloons()
    # Sign in
    user = auth.sign_in_with_email_and_password(email, password)
    db.child(user['localId']).child("Handle").set(handle)
    db.child(user['localId']).child("ID").set(user['localId'])
    st.title('Welcome' + handle)
    st.info('Login via login drop down selection')

# Login Block
if choice == 'Login':
  login = st.sidebar.checkbox('Login')
  if login:
    user = auth.sign_in_with_email_and_password(email, password)
    st.write('<style>div.row-widget.stRadio > div{flex-direction:row;}</style>', unsafe_allow_html=True)
    #bio = st.radio('Jump to', ['Home', 'Workplace Feeds', 'Settings'])


    anu.title("Stock price prediction App")
    ticker = anu.text_input("ticker")
    start_date = anu.date_input("start date")
    end_date = anu.date_input("end date")
    #predict = anu.sidebar.anu.slider('Years of prediction:', 1, 4)
    #period = int(predict) * 365

    data = yf.download(ticker, start=start_date, end=end_date)
    fig = px.line(data, x=data.index, y=data['Adj Close'], title=ticker)
    anu.plotly_chart(fig)
    #anu.write(start_date)

    pricing_data, fundamental_data, news, prediction ,about= streamlit.tabs(
        ["Pricing data", "fundamental data", "top news", "prediction","About"])

    with pricing_data:
        anu.header("price movement")
        #anu.write(data)
        data2=data
        data2['% Change']=data['Adj Close']/data['Adj Close'].shift(1)-1
        data2.dropna(inplace=True)
        anu.write(data2)
        annual_return=data2['% Change'].mean()*252*100
        anu.write('Annual return is',annual_return,'%')
        stdev=np.std(data2['% Change'])*np.sqrt(252)
        anu.write('Standard deviation is ',stdev*100,' %')
        anu.write('Risk adjusted return is ',annual_return/(stdev*100))

    from alpha_vantage.fundamentaldata import FundamentalData
    with fundamental_data:
        anu.header("fundamental data")
        key = "0UE1KZH9OCJZ052Y"
        fd=FundamentalData(key,output_format='pandas')
        anu.subheader('balance sheet')
        balance_sheet=fd.get_balance_sheet_annual(ticker)[0]
        bs=balance_sheet.T[2:]
        bs.columns=list(balance_sheet.T.iloc[0])
        anu.write(bs)
        anu.subheader('income sheet')
        income_statement=fd.get_income_statement_annual(ticker)[0]
        is1=income_statement.T[2:]
        is1.columns=list(income_statement.T.iloc[0])
        anu.write(is1)
        anu.subheader('cash overflow statement')
        cash_flow=fd.get_cash_flow_annual(ticker)[0]
        cf=cash_flow.T[2:]
        cf.columns=list(cash_flow.T.iloc[0])
        anu.write(cf)


    from stocknews import StockNews
    with news:
        anu.header("top news")
        anu.header(f'news of {ticker}')
        sn=StockNews(ticker,save_news=False)
        df_news=sn.read_rss()
        for i in range(10):
            anu.subheader(f'News{i+1}')
            anu.write(df_news['published'][i])
            anu.write(df_news['title'][i])
            anu.write(df_news['summary'][i])
            anu.write(df_news['sentiment_title'][i])
            title_sentiment=df_news['sentiment_title'][i]
            anu.write(f'title sentiment{title_sentiment}')
            news_sentiment=df_news['sentiment_summary'][i]
            anu.write(f'new sentiment{news_sentiment}')


    # from fbprophet import Prophet
    with prediction:
        anu.header("forecasting")
        n_years = anu.slider('Years of prediction:', 1, 4)
        period = n_years * 365

        data.reset_index(inplace=True)
        #anu.write(data.columns)
        df_train = data[['Date','Close']]
        #anu.write(df_train)
        df_train=df_train.rename(columns={"Date":"ds","Close":"y"})

        m=Prophet()
        m.fit(df_train)
        future=m.make_future_dataframe(periods=period)
        forecast=m.predict(future)

        anu.subheader('forecast data')
        anu.write(forecast.tail())
        fig1=plot_plotly(m,forecast)
        anu.plotly_chart(fig1)

        anu.write("Forecast components")
        fig2 = m.plot_components(forecast)
        anu.write(fig2)

    with about:
        anu.write("api used in this projects are alpha vantage and yfinance")
        anu.write('plently of other python modules is used for working of this project')
