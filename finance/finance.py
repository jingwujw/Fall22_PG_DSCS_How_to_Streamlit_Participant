# Adapted from: https://www.kaggle.com/code/ryanholbrook/forecasting-with-machine-learning/tutorial


import plotly.graph_objects as go
from plotly.colors import n_colors

import pandas as pd
from sklearn.linear_model import LinearRegression, MultiTaskLasso
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.multioutput import RegressorChain
from xgboost import XGBRegressor

def create_visualization(y_test:pd.DataFrame, y_pred: pd.DataFrame) -> go.Figure:
    """Creating and returning visualizations for the fitting and prediction data

    Args:
        y_test (pd.DataFrame): Y-test data
        y_pred (pd.DataFrame): Y-prediction data

    Returns:
        go.Figure: Vigure showing predictions and actuals
    """
    fig = go.Figure()

    colorscale = n_colors('rgb(67, 198, 172)', 'rgb(25, 22, 84)', len(y_pred), colortype='rgb')

    for i, (index, row) in enumerate(y_pred.iterrows()):
        fig.add_trace(go.Scatter(x=pd.period_range(start=index, periods=len(row)).to_timestamp(), y=row, line=dict(color=colorscale[i])))
    fig.add_trace(go.Scatter(x=y_test.index, y=y_test.y_step_1, line=dict(color='black')))
    fig.update_layout(showlegend=False)
    
    return fig


def make_lags(ts, lags, lead_time=1):
    """Creating lags for each prediction

    Args:
        ts (_type_): y-data to be 'lagged'
        lags (_type_): Number of lags
        lead_time (int, optional): Lead time aka. first ts in the future to be predicted. Defaults to 1.

    Returns:
        _type_: Lagged data
    """
    return pd.concat(
        {
            f'y_lag_{i}': ts.shift(i)
            for i in range(lead_time, lags + lead_time)
        },
        axis=1)


def make_multistep_target(ts, steps):
    """_summary_

    Args:
        ts (_type_): y-data to be stepped
        steps (_type_): number of steps

    Returns:
        _type_: Multistepped data
    """
    return pd.concat(
        {f'y_step_{i + 1}': ts.shift(-i)
         for i in range(steps)},
        axis=1)


apple_stock = pd.read_csv("./finance/apple.csv", index_col='Date')
apple_stock.index = pd.to_datetime(apple_stock.index)
apple_stock.sort_index(inplace=True)

# Thirty days of lag features
y = apple_stock.Close.copy()
X = make_lags(y, lags=30).fillna(0.0)

# 7 Day forecast
y = make_multistep_target(y, steps=7).dropna()

# Shifting has created indexes that don't match. Only keep times for
# which we have both targets and features.
y, X = y.align(X, join='inner', axis=0)

# Create splits
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, shuffle=False)


### Model 1: Basic Direct LinReg
model = LinearRegression()
model.fit(X_train, y_train)

y_fit = pd.DataFrame(model.predict(X_train), index=X_train.index, columns=y.columns)
y_pred = pd.DataFrame(model.predict(X_test), index=X_test.index, columns=y.columns)

train_rmse = mean_squared_error(y_train, y_fit, squared=False)
test_rmse = mean_squared_error(y_test, y_pred, squared=False)
print((f"Train RMSE: {train_rmse:.2f}\n" f"Test RMSE: {test_rmse:.2f}"))

create_visualization(y_train, y_fit).show()
create_visualization(y_test, y_pred).show()



### Model 3: DirRec XGBoost
from sklearn.multioutput import RegressorChain

model = RegressorChain(XGBRegressor())
model.fit(X_train, y_train)

y_fit = pd.DataFrame(model.predict(X_train), index=X_train.index, columns=y.columns)
y_pred = pd.DataFrame(model.predict(X_test), index=X_test.index, columns=y.columns)

train_rmse = mean_squared_error(y_train, y_fit, squared=False)
test_rmse = mean_squared_error(y_test, y_pred, squared=False)
print((f"Train RMSE: {train_rmse:.2f}\n" f"Test RMSE: {test_rmse:.2f}"))

create_visualization(y_train, y_fit).show()
create_visualization(y_test, y_pred).show()

# How to use XGBoost for stock prediction: https://www.kaggle.com/code/mtszkw/xgboost-for-stock-trend-prices-prediction