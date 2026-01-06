#import necessary libraries
import streamlit as st
import pandas as pd
import numpy as np
import os
import seaborn as sns
import matplotlib.pyplot as plt
import requests
from datetime import datetime
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

#logger
def log(message):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] {message}")

#session state initialization
if "cleaned_saved" not in st.session_state:
    st.session_state.cleaned_saved = False

#folder setup
base_dir = os.path.dirname(os.path.abspath(__file__))
raw_dir = os.path.join(base_dir, "data", "raw")
clean_dir = os.path.join(base_dir, "data", "cleaned")

os.makedirs(raw_dir, exist_ok=True)
os.makedirs(clean_dir, exist_ok=True)

log("application started")
log(f"raw_dir = {raw_dir}")
log(f"clean_dir = {clean_dir}")

#page config
st.set_page_config("End-to-End SVM", layout = "wide")
st.title("End-to-End SVM Classifier Application")

#sidabar : model settings
st.sidebar.header("SVM Settings")
kernel = st.sidebar.selectbox("Kernel", ["linear", "rbf", "poly", "sigmoid"])
C = st.sidebar.slider("C (Regularization)", 0.01, 10.0, 1.0)
gamma = st.sidebar.selectbox("Gamma", ["scale", "auto"])

log(f"SVM settings - kernel: {kernel}, C: {C}, gamma: {gamma}")

#step 1 : Data Ingestion
st.header("Step 1: Data Ingestion")
log("Step 1: Data Ingestion started")

option = st.radio("Choose Data Source", ["Download Dataset", "Upload CSV"])
df = None
raw_path = None

if option == "Download Dataset":
    if st.button("Download Iris Dataset"):
        log("Downloading Iris dataset")
        url = "https://raw.githubusercontent.com/mwaskom/seaborn-data/master/iris.csv"
        response = requests.get(url)

        raw_path = os.path.join(raw_dir, "iris.csv")
        with open(raw_path, "wb") as f:
            f.write(response.content)

        df = pd.read_csv(raw_path)
        st.success("Dataset Downloaded successfully")
        log(f"Iris dataset saved at {raw_path}")

if option == "Upload CSV":
    uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])
    if uploaded_file:
        raw_path = os.path.join(raw_dir, uploaded_file.name)
        with open(raw_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        df = pd.read_csv(raw_path)
        st.success("File uploaded successfully")
        log(f"Uploaded file saved at {raw_path}")

#step 2 : EDA
if df is not None:
    st.header("Step 2: Exploratory Data Analysis (EDA)")
    log("Step 2: EDA started")

    st.dataframe(df.head())
    st.write("Shape", df.shape)
    st.write("Missing Values", df.isnull().sum())

    fig, ax = plt.subplots()
    sns.heatmap(df.corr(numeric_only = True), annot = True, cmap = "coolwarm", ax = ax)
    st.pyplot(fig)

    log("EDA completed")

#step 3 : Data Cleaning
if df is not None:
    st.header("Step 3 : Data Cleaning")
    strategy = st.selectbox(
        "Missing Value Handling Strategy",
        ["Mean", "Median", "Drop Rows"]
    )
    df_clean = df.copy()
    if strategy == "Drop Rows":
        df_clean = df_clean.dropna()
    else:
        for col in df_clean.select_dtypes(include=[np.number]).columns:
            if strategy == "Mean":
                df_clean[col].fillna(df_clean[col].mean(), inplace=True)
            elif strategy == "Median":
                df_clean[col].fillna(df_clean[col].median(), inplace=True)
    st.session_state.df_clean = df_clean
    st.success("Data Cleaning Completed")
else:
    st.info("Please complete Step 1 to proceed.")

#step 4 : Model Training
if st.button("Save cleaned dataset"):
    if st.session_state.df_clean is None:
        st.error("No cleaned data to save. Please complete Step 3.")
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        clean_filename = f"cleaned_iris_{timestamp}.csv"
        clean_path = os.path.join(clean_dir, clean_filename)

        st.session_state.df_clean.to_csv(clean_path, index=False)
        st.success("cleaned dataset saved successfully")
        st.info(f"Cleaned dataset saved at {clean_path}")
        log(f"Cleaned dataset saved at {clean_path}")
    
#step 5 : Load cleaned dataset
st.header("step 5 : load cleaned dataset")
clean_files = os.listdir(clean_dir)
if not clean_files:
    st.warning("no cleaned datasets found. please save one in step 4")
    log("no cleaned datasets found. please save one in step 4")
else:
    selected = st.selectbox("Select cleaned dataset", clean_files)
    df_model = pd.read_csv(os.path.join(clean_dir, selected))
    st.success(f"Loaded dataset: {selected}")
    log(f"Loaded cleaned dataset: {selected}")
    
    st.dataframe(df_model.head())

#step 6 : train svm
st.header("step 6 : train svm")
log("step 6 : train svm started")

target = st.selectbox("Select target variable", df_model.columns)
y = df_model[target]
if y.dtype == 'object':
    y = LabelEncoder().fit_transform(y)
    log("target column encoded")

#select numerical features only
x = df_model.drop(columns=[target])
x = x.select_dtypes(include=[np.number])
if x.empty:
    st.error("No numerical features available for training.")
    st.stop()

#scale features
scaler = StandardScaler()
x = scaler.fit_transform(x)

#train - test split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.25, random_state = 42)

#model initialization
model = SVC(kernel = kernel, C = C, gamma = gamma)
model.fit(x_train, y_train)
st.success("SVM model trained successfully")
log("SVM model trained successfully")

#evaluation metrics
y_pred = model.predict(x_test)
acc = accuracy_score(y_test, y_pred)
st.success(f"Model Accuracy: {acc:.4f}")
log(f"Model Accuracy: {acc:.4f}")

cm = confusion_matrix(y_test, y_pred)
fig, ax = plt.subplots()
sns.heatmap(cm, annot = True, fmt = 'd', cmap = 'Blues', ax = ax)
st.pyplot(fig)
log("Confusion matrix displayed")