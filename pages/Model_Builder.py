import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

st.title("Model Builder")

# Check if dataset is loaded
if "df" not in st.session_state:
    st.warning("Please upload or load a dataset first!")
    st.stop()

df = st.session_state.df

# --- Target Selection ---
target_column = st.selectbox("Choose target column", df.columns)

# --- Feature Selection with Preview ---
feature_columns = st.multiselect(
    "Choose feature columns",
    [col for col in df.columns if col != target_column]
)

if feature_columns:
    st.write("**Selected Features Preview:**")
    st.dataframe(df[feature_columns + [target_column]].head(10))

# After you load your dataset in model_builder.py
st.subheader("Dataset Preview")
st.dataframe(df.head())  # Show first 5 rows


# --- Sidebar Model Settings ---
st.sidebar.header("Model Settings")
n_estimators = st.sidebar.slider("Number of Trees", 10, 500, 100)
max_depth = st.sidebar.slider("Max Depth", 2, 20, 5)

# --- Train Model ---
if st.button("Train Model"):
    X = df[feature_columns]
    y = df[target_column]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=42
    )
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # --- Results ---
    st.subheader("Results")
    st.write(f"Accuracy: {accuracy_score(y_test, y_pred):.2f}")
    st.text("Classification Report:")
    st.text(classification_report(y_test, y_pred))
