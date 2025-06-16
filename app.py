import streamlit as st
import pandas as pd
import os

st.title("Noctem's Dashboard")

sourceSelect = st.radio("Choose source", ["Kaggle Dataset", "Upload Dataset"])

if sourceSelect == "Kaggle dataset":
    pass