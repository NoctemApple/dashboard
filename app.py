import streamlit as st
import pandas as pd
import os
import subprocess
import zipfile
import glob
import kaggle
from kaggle.api.kaggle_api_extended import KaggleApi
import matplotlib.pyplot as plt
import seaborn as sns

# Kaggle check

KAGGLE_REPO_JSON = "kaggle.json"
if os.path.exists(KAGGLE_REPO_JSON):
    st.success("Found kaggle.json")
    os.environ["KAGGLE_CONFIG_DIR"] = "."
else:
    st.warning("No kaggle.json found")

try:
    api = KaggleApi()
    api.authenticate()
except Exception as e:
    st.error(f"Kaggle API failed to Authenticate: {e}")

# UI Starts Here

st.title("Noctem's Dashboard")
st.markdown("Upload a dataset or Paste a Kaggle Link")
sourceSelect = st.radio("Choose source", ["Kaggle Dataset", "Upload Dataset"])

os.makedirs("data", exist_ok=True)

if sourceSelect == "Kaggle Dataset":
    kaggle_url = st.text_input("Insert link here:",
                               placeholder="https://www.kaggle.com/datasets/blastchar/telco-customer-churn")

    if st.button("Download") and kaggle_url.strip():
        try:
            parts = kaggle_url.strip().split("/")
            slug = f"{parts[-2]}/{parts[-1]}" if "datasets" in parts else None

            if not slug:
                st.error("Invalid Kaggle dataset URL")
            else:
                st.info(f"Downloading dataset `{slug}`...")
                output_path = "data/kaggle_dataset.zip"

                api.dataset_download_files(slug, path="data", quiet=False)

                zip_files = glob.glob("data/*.zip")
                if not zip_files:
                    st.error("No zip file downloaded")
                else:
                    zip_path = zip_files[0]

                    with zipfile.ZipFile(zip_path, "r") as zip_ref:
                        zip_ref.extractall("data")
                    os.remove(zip_path)

                    st.success("Dataset downloaded and extracted")

        except Exception as e:
            st.error(f"error in getting Kaggle dataset: {e}")

    csv_files = glob.glob("data/*.csv")
    if csv_files:
        selected_csv = st.selectbox("Choose a CSV file",[os.path.basename(f) for f in csv_files])
        
        if st.button("Load Selected CSV"):
            st.session_state.selected_csv_to_load = selected_csv
        
    if "selected_csv_to_load" in st.session_state:
        filepath = os.path.join("data", st.session_state.selected_csv_to_load)
        if os.path.exists(filepath):
            df = pd.read_csv(filepath)
            st.session_state.df = df
            st.session_state.filename = st.session_state.selected_csv_to_load
            st.success(f"Loaded: {st.session_state.selected_csv_to_load}")
        else:
            st.warning("Selected file no longer exists")
    
    elif not csv_files:
        st.warning("No CSV file found")

elif sourceSelect == "Upload Dataset":
    file = st.file_uploader("Upload CSV file", type="csv")
    if file is not None:
        df = pd.read_csv(file)
        st.session_state.df = df
        st.session_state.filename = file.name
        st.success(f"CSV Uploaded: {file.name}")

# Features start here

if 'df' in st.session_state:
    df = st.session_state.df
    filename = st.session_state.get("filename", "Uploaded Data")

    st.subheader("Dataset Overview")
    st.write(f"**Filename:** `{filename}`")
    st.write(f"**Shape**: {df.shape[0]} rows x {df.shape[1]} columns")
    st.write(f"**Memory Usage:** {df.memory_usage(deep=True).sum() / 1024:.2f} KB")

    st.subheader("Column Info")
    col_info = pd.DataFrame({
        "Column": df.columns,
        "Data Type": df.dtypes,
        "Null Values": df.isnull().sum(),
        "Unique Values": df.nunique()
    })
    st.dataframe(col_info)

    st.subheader("Data Preview")
    columns = st.multiselect("Select columns to view", df.columns.tolist(), default=df.columns.tolist()[:5])
    
    if columns:
        st.dataframe(df[columns].head(10))
    else:
        st.info("No columns selected")

    st.subheader("Descriptive Statistics")
    st.dataframe(df.describe().T)

    st.subheader("Missing Data Map")
    fig, ax = plt.subplots()
    sns.heatmap(df.isnull().iloc[:, :30], cbar=False, cmap="viridis", ax=ax)
    st.pyplot(fig)

    st.subheader("Quick Charts")
    selected_col = st.selectbox("Pick a column to plot", df.columns)
    if pd.api.types.is_numeric_dtype(df[selected_col]):
        st.bar_chart(df[selected_col].value_counts().sort_index())
    else:
        st.bar_chart(df[selected_col].value_counts().head(10))

else:
    st.info("Upload or download a dataset first")

