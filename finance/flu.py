# Adapted from: https://www.kaggle.com/code/ryanholbrook/forecasting-with-machine-learning/tutorial
from pathlib import Path

import plotly.graph_objects as go
from plotly.colors import n_colors

import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
from joblib import dump, load

def create_visualization(y_test:pd.DataFrame, y_pred: pd.DataFrame) -> go.Figure:
    y_pred.index=y_pred.index.to_timestamp()

    fig = go.Figure()

    colorscale = n_colors('rgb(67, 198, 172)', 'rgb(25, 22, 84)', len(y_pred), colortype='rgb')

    for i, (index, row) in enumerate(y_pred.iterrows()):
        fig.add_trace(go.Scatter(x=pd.period_range(start=index, periods=len(row)).to_timestamp(), y=row, line=dict(color=colorscale[i])))
    fig.add_trace(go.Scatter(x=y_test.index, y=y_test.y_step_1))
    fig.update_layout(showlegend=False)
    
    return fig


data_dir = Path("./finance")
flu_trends = pd.read_csv(data_dir / "flu-trends.csv")
flu_trends.set_index(
    pd.PeriodIndex(flu_trends.Week, freq="W"),
    inplace=True,
)
flu_trends.drop("Week", axis=1, inplace=True)

def make_lags(ts, lags, lead_time=1):
    return pd.concat(
        {
            f'y_lag_{i}': ts.shift(i)
            for i in range(lead_time, lags + lead_time)
        },
        axis=1)


# Four weeks of lag features
y = flu_trends.FluVisits.copy()
X = make_lags(y, lags=4).fillna(0.0)


def make_multistep_target(ts, steps):
    return pd.concat(
        {f'y_step_{i + 1}': ts.shift(-i)
         for i in range(steps)},
        axis=1)


# Eight-week forecast
y = make_multistep_target(y, steps=8).dropna()

# Shifting has created indexes that don't match. Only keep times for
# which we have both targets and features.
y, X = y.align(X, join='inner', axis=0)

# Create splits
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, shuffle=False)
y_test.index=y_test.index.to_timestamp()
y_train.index=y_train.index.to_timestamp()



### Model 1: Basic Direct LinReg
model = LinearRegression()
model.fit(X_train, y_train)
dump(model, './finance/dir_linreg.joblib')

y_fit = pd.DataFrame(model.predict(X_train), index=X_train.index, columns=y.columns)
y_pred = pd.DataFrame(model.predict(X_test), index=X_test.index, columns=y.columns)

train_rmse = mean_squared_error(y_train, y_fit, squared=False)
test_rmse = mean_squared_error(y_test, y_pred, squared=False)
print((f"Train RMSE: {train_rmse:.2f}\n" f"Test RMSE: {test_rmse:.2f}"))

create_visualization(y_test, y_pred).show()
create_visualization(y_train, y_fit).show()


### Model 2: Direct XGBoost
from sklearn.multioutput import MultiOutputRegressor

model = MultiOutputRegressor(XGBRegressor())
model.fit(X_train, y_train)
dump(model, './finance/dir_xgb.joblib')

y_fit = pd.DataFrame(model.predict(X_train), index=X_train.index, columns=y.columns)
y_pred = pd.DataFrame(model.predict(X_test), index=X_test.index, columns=y.columns)

train_rmse = mean_squared_error(y_train, y_fit, squared=False)
test_rmse = mean_squared_error(y_test, y_pred, squared=False)
print((f"Train RMSE: {train_rmse:.2f}\n" f"Test RMSE: {test_rmse:.2f}"))

create_visualization(y_test, y_pred).show()


### Model 3: DirRec XGBoost
from sklearn.multioutput import RegressorChain

model = RegressorChain(XGBRegressor())
model.fit(X_train, y_train)
dump(model, './finance/dirrec_xgb.joblib')

y_fit = pd.DataFrame(model.predict(X_train), index=X_train.index, columns=y.columns)
y_pred = pd.DataFrame(model.predict(X_test), index=X_test.index, columns=y.columns)

train_rmse = mean_squared_error(y_train, y_fit, squared=False)
test_rmse = mean_squared_error(y_test, y_pred, squared=False)
print((f"Train RMSE: {train_rmse:.2f}\n" f"Test RMSE: {test_rmse:.2f}"))

create_visualization(y_test, y_pred).show()