import streamlit as st
import pandas as pd
import os
import glob
import matplotlib.pyplot as plt
import seaborn as sns

st.title("Noctem's Dashboard")

# Always ensure data folder exists
os.makedirs("data", exist_ok=True)

st.markdown("Upload a dataset to get started")

if 'df' not in st.session_state:
    csv_files = glob.glob("data/*.csv")
    if csv_files:
        latest = max(csv_files, key=os.path.getctime)
        df = pd.read_csv(latest)
        st.session_state.df = df
        st.session_state.filename = os.path.basename(latest)
        st.session_state.selected_csv_to_load = os.path.basename(latest)
        st.success(f"Previously Loaded: {st.session_state.filename}")

# Upload Dataset
file = st.file_uploader("Upload CSV file", type="csv")
if file is not None:
    # Define save path
    save_path = os.path.join("data", file.name)

    # Save uploaded file permanently in 'data/'
    with open(save_path, "wb") as f:
        f.write(file.getbuffer())

    # Load into DataFrame
    df = pd.read_csv(save_path)
    st.session_state.df = df
    st.session_state.filename = file.name
    st.session_state.selected_csv_to_load = file.name

    st.success(f"CSV uploaded and saved to 'data/' as {file.name}")

# If dataset exists in session
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
        "Data Type": df.dtypes.astype(str),
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
    subset_cols = df.columns[:30] if df.shape[1] > 30 else df.columns
    missing_count = df[subset_cols].isnull().sum().sum()

    if missing_count == 0:
        st.info("No missing values found.")
    else:
        fig, ax = plt.subplots()
        sns.heatmap(df[subset_cols].isnull(), cbar=False, cmap=sns.color_palette(["blue", "red"]), ax=ax)
        st.pyplot(fig)

    st.subheader("Quick Charts")
    selected_col = st.selectbox("Pick a column to plot", df.columns)
    if pd.api.types.is_numeric_dtype(df[selected_col]):
        st.bar_chart(df[selected_col].value_counts().sort_index())
    else:
        st.bar_chart(df[selected_col].value_counts().head(10))

    # Correlation Heatmap
    if df.select_dtypes(include='number').shape[1] > 1:
        st.subheader("Correlation Heatmap")
        fig, ax = plt.subplots()
        sns.heatmap(df.corr(numeric_only=True), annot=True, cmap="coolwarm", ax=ax)
        st.pyplot(fig)

    # Filtering
    if st.checkbox("Enable Filters"):
        filter_col = st.selectbox("Filter Column", df.columns)
        unique_vals = df[filter_col].dropna().unique()
        selected_val = st.selectbox("Select value", unique_vals)
        df_filtered = df[df[filter_col] == selected_val]
        st.write(f"Filtered {len(df_filtered)} rows")
        st.dataframe(df_filtered.head(10))

    # Download cleaned CSV
    st.download_button("Download Cleaned CSV", df.to_csv(index=False), "cleaned_data.csv")

else:
    st.info("Upload a dataset first")

# QoL stuff
st.sidebar.header("Quality of Life Actions")

if st.sidebar.button("Clear Dataset"):
    for key in ['df', 'filename', 'selected_csv_to_load']:
        st.session_state.pop(key, None)
    st.rerun()

if st.sidebar.button("Clear All Files in Data Folder"):
    for f in glob.glob("data/*"):
        os.remove(f)
    st.success("Cleared all files in 'data/'")
