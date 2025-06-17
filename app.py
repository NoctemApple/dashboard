import streamlit as st
import pandas as pd
import os
import subprocess
import zipfile
import glob
import kaggle
from kaggle.api.kaggle_api_extended import KaggleApi

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

    if st.button("Download") and kaggle_url:
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

                    csv_files = glob.glob("data/*.csv")
                    if csv_files:
                        df = pd.read_csv(csv_files[0])
                        st.success(f"Loaded: {os.path.basename(csv_files[0])}")
                        st.dataframe(df.head())
                    else:
                        st.warning("No CSV file found")

        except Exception as e:
            st.error(f"error in getting Kaggle dataset: {e}")

elif sourceSelect == "Upload Dataset":
    file = st.file_uploader("Upload CSV file", type="csv")
    if file is not None:
        df = pd.read_csv(file)
        st.success("CSV Uploaded")
        st.dataframe(df.head())

# Features start here

st.subheader("Dataset Overview")

st.write(f"**Filename:** `{os.path.basename(csv_files[0])}`")
st.write(f"**Shape**:** {df.shape[0]} rows x {df.shape[1]} columns")
st.write(f"**Memory Usage:** {df.memory_usage(deep=True).sum() / 1024:.2f} KB")

st.subheader("Column Info")
col_info = pd.DataFrame({
    "Column": df.columns,
    "Data Type": df.dtypes,
    "Null Values": df.isnull().sum(),
    "Unique Values": df.nunique()
})
st.dataframe(col_info)

