import streamlit as st
import pandas as pd

st.header('Hello, world 👋')

apple_stock = pd.read_csv("./finance/apple.csv", index_col='Date')
apple_stock.index = pd.to_datetime(apple_stock.index)
apple_stock.sort_index(inplace=True)


with st.expander("Raw data"):
    st.write(apple_stock)


st.code("""


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


""")


tab1, tab2, tab3 = st.tabs(["Test1", "Test2", "Test3"])

with tab1:
    st.write("This is tab2.")