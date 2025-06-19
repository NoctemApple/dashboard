import streamlit as st
import pandas as pd
import joblib
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

st.set_page_config(page_title="Model Trainer", layout="wide")
st.title("Train a Model")


if 'df' not in st.session_state:
    st.warning("Please upload or download a dataset from the Home page")
    st.stop()

df = st.session_state.df

st.subheader("Select Target and Features")
target_column = st.selectbox("Choose target column", df.columns)
feature_columns = st.multiselect("Choose feature columns", [col for col in df.columns if col != target_column])

if st.button("Train Model"):
    if not feature_columns:
        st.warning("Select at least one feature column")
    else:
        X = pd.get_dummies(df[feature_columns])
        y = pd.factorize(df[target_column])[0] if df[target_column].dtype == 'object' else df[target_column]

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

        model = RandomForestClassifier()
        model.fit(X_train, y_train)

        accuracy = accuracy_score(y_test, model.predict(X_test))
        st.success(f"Model accuracy: {accuracy:.2%}")

        os.makedirs("models", exist_ok=True)
        model_path = f"models/{target_column}_rf_model.pkl"
        joblib.dump(model,model_path)
        st.info(f"Model saved to `{model_path}`")




