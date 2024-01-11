import pandas as pd
import streamlit as st
import seaborn as sns

st.header('Hi Streamlit')

PATH_DATA = 'data/data_diplom.csv'

@st.cache_data
def load_data(path):
    '''Загрузка данных'''
    data = pd.read_csv(path)
    data = data.sample(5000)
    return data

@st.cache_data
def transform(data):
    colors = sns.color_palette("coolwarm").as_hex()
    n_colors = len(colors)

    data = data.reset_index(drop=True)
    data["norm_price"]  = data["price"] / data["area"]

    data["label_colors"] = pd.qcut(data["norm_price"], n_colors, labels=colors)
    data["label_colors"] = data["label_colors"].astype("str")
    return data

df = load_data(PATH_DATA)
df = transform(df)
st.write(df[:4])

st.map(data=df, latitude='geo_lat', longitude='geo_lon', color='label_colors')
