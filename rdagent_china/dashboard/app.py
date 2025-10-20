import pandas as pd
import plotly.express as px
import streamlit as st

from rdagent_china.db import get_db
from rdagent_china.data.universe import get_csi300_symbols

st.set_page_config(page_title="RD-Agent China Dashboard", layout="wide")

st.title("RD-Agent China Dashboard")

symbols = get_csi300_symbols()[:50]
choice = st.selectbox("Symbol", symbols)

start = st.date_input("Start")
end = st.date_input("End")

if st.button("Load"):
    db = get_db()
    db.init()
    df = db.read_prices(symbols=[choice], start=str(start), end=str(end))
    if df.empty:
        st.warning("No data. Run rdc ingest first.")
    else:
        fig = px.line(df, x="date", y="close", title=f"{choice} Close Price")
        st.plotly_chart(fig, use_container_width=True)
