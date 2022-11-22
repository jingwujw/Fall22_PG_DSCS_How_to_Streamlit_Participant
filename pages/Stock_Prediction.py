# Adapted from: https://www.kaggle.com/code/ryanholbrook/forecasting-with-machine-learning/tutorial
import streamlit as st

import plotly.graph_objects as go
from plotly.colors import n_colors

import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.multioutput import RegressorChain
from xgboost import XGBRegressor

def create_visualization(y_test:pd.DataFrame, y_pred: pd.DataFrame) -> go.Figure:
    fig = go.Figure()

    colorscale = n_colors('rgb(67, 198, 172)', 'rgb(25, 22, 84)', len(y_pred), colortype='rgb')

    for i, (index, row) in enumerate(y_pred.iterrows()):
        fig.add_trace(go.Scatter(x=pd.period_range(start=index, periods=len(row)).to_timestamp(), y=row, line=dict(color=colorscale[i])))
    fig.add_trace(go.Scatter(x=y_test.index, y=y_test.y_step_1, line=dict(color='black')))
    fig.update_layout(showlegend=False)
    
    return fig


def make_lags(ts, lags, lead_time=1):
    return pd.concat(
        {
            f'y_lag_{i}': ts.shift(i)
            for i in range(lead_time, lags + lead_time)
        },
        axis=1)


def make_multistep_target(ts, steps):
    return pd.concat(
        {f'y_step_{i + 1}': ts.shift(-i)
         for i in range(steps)},
        axis=1)

st.title('Stock Prediction Project')

st.header('Motivation')
st.write('I like finance and I like machine learning so I started to work on this project.')

st.header('The Data')
st.markdown('''
For this project, I decided to take stock market data for Apple, Inc. $APPL and apply different machine learnig algorithms to predict the closing price.

[Source](https://www.kaggle.com/datasets/tarunpaparaju/apple-aapl-historical-stock-data)

Here is a quick overview of the dataset:
''')
apple_stock = pd.read_csv("./finance/apple.csv", index_col='Date')
apple_stock.index = pd.to_datetime(apple_stock.index)
apple_stock.sort_index(inplace=True)

with st.expander('Original Dataset'):
    st.dataframe(apple_stock)

st.write('All of the predictions are going to be made based on the last 30 day closing history:')
# Thirty days of lag features
y = apple_stock.Close.copy()
X = make_lags(y, lags=30).fillna(0.0)
with st.expander('Dataset with Lag'):
    st.dataframe(X)

st.write('And for each timestep, I am predicting 7 days into the future:')
# 7 Day forecast
y = make_multistep_target(y, steps=7).dropna()
with st.expander('Dataset Predictions'):
    st.dataframe(y)

# Shifting has created indexes that don't match. Only keep times for
# which we have both targets and features.
y, X = y.align(X, join='inner', axis=0)

st.write('In this case, I used a test size of 25% with no shuffle to preserve the timeseries characteristics of the dataset.')
# Create splits
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, shuffle=False)

st.header('Machine Learning Approaches')
tab_1, tab_2 = st.tabs(['Linear Regression', 'XGBoost Regressor'])
with tab_1:
    st.markdown('''
    ### Definition
    LinearRegression fits a linear model with coefficients w = (w1, …, wp) to minimize the residual sum of squares between the observed targets in the dataset, and the targets predicted by the linear approximation.
    [Source](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html)

    I used this model because...
    ''')
    ### Model 1: Basic Direct LinReg
    model = LinearRegression()
    model.fit(X_train, y_train)

    y_fit = pd.DataFrame(model.predict(X_train), index=X_train.index, columns=y.columns)
    y_pred = pd.DataFrame(model.predict(X_test), index=X_test.index, columns=y.columns)

    train_rmse = mean_squared_error(y_train, y_fit, squared=False)
    test_rmse = mean_squared_error(y_test, y_pred, squared=False)

    st.subheader('Overview training & fit:')
    st.plotly_chart(create_visualization(y_train, y_fit))

    st.subheader('Overview predictions:')
    st.plotly_chart(create_visualization(y_test, y_pred))

    col1, col2 = st.columns(2)
    col1.metric('Train RMSE', round(train_rmse,2))
    col2.metric('Test RMSE', round(test_rmse,2))

    st.markdown('''
    ### Analysis

    Linreg works kinda well.
    ''')

with tab_2:
    st.markdown('''
    ### Definition
    Regression predictive modeling problems involve predicting a numerical value such as a dollar amount or a height. XGBoost can be used directly for regression predictive modeling.
    [Source](https://machinelearningmastery.com/xgboost-for-regression/)

    I used this model because...
    ''')
    ### Model 2: DirRec XGBoost

    model = RegressorChain(XGBRegressor())
    model.fit(X_train, y_train)

    y_fit = pd.DataFrame(model.predict(X_train), index=X_train.index, columns=y.columns)
    y_pred = pd.DataFrame(model.predict(X_test), index=X_test.index, columns=y.columns)

    train_rmse = mean_squared_error(y_train, y_fit, squared=False)
    test_rmse = mean_squared_error(y_test, y_pred, squared=False)

    st.subheader('Overview training & fit:')
    st.plotly_chart(create_visualization(y_train, y_fit))

    st.subheader('Overview predictions:')
    st.plotly_chart(create_visualization(y_test, y_pred))

    col1, col2 = st.columns(2)
    col1.metric('Train RMSE', round(train_rmse,2))
    col2.metric('Test RMSE', round(test_rmse,2))

    st.markdown('''
    ### Analysis

    XGBoostRegressor seems to be overfittig a lot and not generalize very well!
    I'll probably need to add additoinal features!
    ''')

st.header('Conclusion')
st.markdown('''
This is my conclusion!
''')

# How to use XGBoost for stock prediction: https://www.kaggle.com/code/mtszkw/xgboost-for-stock-trend-prices-prediction


