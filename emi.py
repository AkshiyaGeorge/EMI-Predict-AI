# 📦 Imports
import streamlit as st
import pandas as pd
import numpy as np
import pickle
import seaborn as sns
import matplotlib.pyplot as plt
import os
import mlflow
from PIL import Image

# ✅ Load models
with open("models/classifier.pkl", "rb") as f:
    classifier = pickle.load(f)

with open("models/regressor.pkl", "rb") as f:
    regressor = pickle.load(f)

with open("models/classifier_simple.pkl", "rb") as f:
    simple_classifier = pickle.load(f)

with open("models/regressor_simple.pkl", "rb") as f:
    regressor_simple = pickle.load(f)

# ✅ Load dataset
df = pd.read_csv("emi_model_data.csv")


# ✅ Page setup
import streamlit as st
st.set_page_config(page_title="EMI Prediction Dashboard", layout="wide")

# ✅ Styled header and welcome message (fully centered)
st.markdown("""
    <div style='text-align: center; padding-top: 10px;'>
        <h1>🏠 EMI Prediction Dashboard</h1>
    </div>
""", unsafe_allow_html=True)

# ✅ Tab layout (full-width, placed directly below centered header)
left_pad, tab_area, right_pad = st.columns([2, 6, 1])
with tab_area:

    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "🏠 Home", 
        "📊 Data Explorer", 
        "✅ EMI Eligibility Predictor",
        "💰 EMI Amount Predictor", 
        "📈 Model Monitoring", 
        "🛠 Admin Panel"
    ])



# 🏠 Tab 1: Home (optional repeat of welcome message if needed)
with tab1:
    st.markdown("""
        <div style='text-align: center; font-size: 18px; padding-top: 20px;'>
            Welcome to your EMI prediction platform.<br>
            Use the tabs above to explore data, make predictions, monitor models, and manage datasets.
        </div>
    """, unsafe_allow_html=True)



# 📊 Tab 2: Data Explorer
with tab2:
    st.title("📊 Data Explorer")
    st.dataframe(df.head())

    st.subheader("Credit Score Distribution")
    fig, ax = plt.subplots()
    sns.histplot(df["credit_score"], bins=30, ax=ax)
    st.pyplot(fig)

    st.subheader("Correlation Heatmap")
    fig, ax = plt.subplots()
    sns.heatmap(df.corr(numeric_only=True), cmap="coolwarm", ax=ax)
    st.pyplot(fig)

    st.subheader("Monthly Salary Distribution")
    fig, ax = plt.subplots()
    sns.histplot(df["monthly_salary"], bins=30, kde=True, color="skyblue", ax=ax)
    st.pyplot(fig)

    st.subheader("Max EMI vs Credit Score")
    fig, ax = plt.subplots()
    sns.scatterplot(data=df, x="credit_score", y="max_monthly_emi", hue="gender", ax=ax)
    st.pyplot(fig)

    st.subheader("Max EMI vs Requested Tenure")
    fig, ax = plt.subplots()
    sns.lineplot(data=df, x="requested_tenure", y="max_monthly_emi", ax=ax)
    st.pyplot(fig)

    st.subheader("Applicant Count by Gender")
    fig, ax = plt.subplots()
    sns.countplot(data=df, x="gender", palette="pastel", ax=ax)
    st.pyplot(fig)

    st.subheader("Current EMI by Existing Loans")
    fig, ax = plt.subplots()
    sns.boxplot(data=df, x="existing_loans", y="current_emi_amount", palette="Set2", ax=ax)
    st.pyplot(fig)




# ✅ Tab 3: EMI Eligibility Predictor
with tab3:
    st.title("✅ EMI Eligibility Predictor")

    gender = st.selectbox("Gender", ["Male", "Female"], key="gender_tab3")
    employment_type = st.selectbox("Employment Type", ["Salaried", "Self-Employed", "Freelancer"], key="employment_tab3")
    monthly_salary = st.number_input("Monthly Salary", min_value=0, key="salary_tab3")
    credit_score = st.slider("Credit Score", 300, 900, key="credit_tab3")
    existing_loans = st.selectbox("Existing Loans", ["Yes", "No"], key="loans_tab3")
    monthly_rent = st.number_input("Monthly Rent", min_value=0, key="rent_tab3")
    school_fees = st.number_input("School Fees", min_value=0, key="school_tab3")
    college_fees = st.number_input("College Fees", min_value=0, key="college_tab3")
    dependents = st.number_input("Number of Dependents", min_value=0, key="dependents_tab3")
    other_expenses = st.number_input("Other Monthly Expenses", min_value=0, key="expenses_tab3")
    requested_amount = st.number_input("Requested Loan Amount", min_value=0, key="amount_tab3")
    requested_tenure = st.slider("Requested Tenure (months)", 1, 120, key="tenure_tab3")

    input_df = pd.DataFrame({
        "gender": [gender],
        "employment_type": [employment_type],
        "monthly_salary": [monthly_salary],
        "credit_score": [credit_score],
        "existing_loans": [existing_loans],
        "monthly_rent": [monthly_rent],
        "school_fees": [school_fees],
        "college_fees": [college_fees],
        "dependents": [dependents],
        "other_monthly_expenses": [other_expenses],
        "requested_amount": [requested_amount],
        "requested_tenure": [requested_tenure]
    })

    def preprocess_input(df):
        df = df.copy()
        df["gender"] = df["gender"].map({"Male": 0, "Female": 1})
        df["existing_loans"] = df["existing_loans"].map({"No": 0, "Yes": 1})
        df["employment_type"] = df["employment_type"].map({
            "Salaried": 0, "Self-Employed": 1, "Freelancer": 2
        })
        return df

    processed_df = preprocess_input(input_df)
    ordered_columns = [
        "gender", "monthly_salary", "credit_score", "existing_loans",
        "monthly_rent", "school_fees", "college_fees", "dependents",
        "other_monthly_expenses", "requested_amount", "requested_tenure",
        "employment_type"
    ]
    processed_df = processed_df[ordered_columns]

    if st.button("Predict Eligibility", key="predict_tab3"):
        prediction = simple_classifier.predict(processed_df)[0]
        st.success("✅ Eligible" if prediction == 1 else "❌ Not Eligible")


# 💰 Tab 4: EMI Amount Predictor
with tab4:
    st.title("💰 EMI Amount Predictor")

    # 🔹 User Inputs with unique keys
    gender = st.selectbox("Gender", ["Male", "Female"], key="gender_tab4")
    employment_type = st.selectbox("Employment Type", ["Salaried", "Self-Employed", "Freelancer"], key="employment_tab4")
    monthly_salary = st.number_input("Monthly Salary", min_value=0, key="salary_tab4")
    credit_score = st.slider("Credit Score", 300, 900, key="credit_tab4")
    existing_loans = st.selectbox("Existing Loans", ["Yes", "No"], key="loans_tab4")
    monthly_rent = st.number_input("Monthly Rent", min_value=0, key="rent_tab4")
    school_fees = st.number_input("School Fees", min_value=0, key="school_tab4")
    college_fees = st.number_input("College Fees", min_value=0, key="college_tab4")
    dependents = st.number_input("Number of Dependents", min_value=0, key="dependents_tab4")
    other_expenses = st.number_input("Other Monthly Expenses", min_value=0, key="expenses_tab4")
    requested_amount = st.number_input("Requested Loan Amount", min_value=0, key="amount_tab4")
    requested_tenure = st.slider("Requested Tenure (months)", 1, 120, key="tenure_tab4")

    # 🔹 Raw Input DataFrame
    input_df = pd.DataFrame({
        "gender": [gender],
        "employment_type": [employment_type],
        "monthly_salary": [monthly_salary],
        "credit_score": [credit_score],
        "existing_loans": [existing_loans],
        "monthly_rent": [monthly_rent],
        "school_fees": [school_fees],
        "college_fees": [college_fees],
        "dependents": [dependents],
        "other_monthly_expenses": [other_expenses],
        "requested_amount": [requested_amount],
        "requested_tenure": [requested_tenure]
    })

    # 🔧 Preprocessing
    def preprocess_input(df):
        df = df.copy()
        df["gender"] = df["gender"].map({"Male": 0, "Female": 1})
        df["existing_loans"] = df["existing_loans"].map({"No": 0, "Yes": 1})
        df["employment_type"] = df["employment_type"].map({
            "Salaried": 0, "Self-Employed": 1, "Freelancer": 2
        })
        return df

    processed_df = preprocess_input(input_df)

    # ✅ Ensure column order matches training
    ordered_columns = [
        "gender", "monthly_salary", "credit_score", "existing_loans",
        "monthly_rent", "school_fees", "college_fees", "dependents",
        "other_monthly_expenses", "requested_amount", "requested_tenure",
        "employment_type"
    ]
    processed_df = processed_df[ordered_columns]

    # ✅ Load the correct model for 12 features
    with open("models/regressor_simple.pkl", "rb") as f:
        regressor = pickle.load(f)

    # 🔮 Predict
    if st.button("Predict EMI", key="predict_tab4"):
        prediction = regressor.predict(processed_df)[0]
        st.success(f"Estimated EMI: ₹{prediction:.2f}")


# 📈 Tab 5: Model Monitoring
with tab5:
    st.title("📈 Model Monitoring")

    mlflow.set_tracking_uri("http://localhost:5000")
    experiment = mlflow.get_experiment_by_name("EMI_Prediction_Models")
    experiment_id = experiment.experiment_id
    runs_df = mlflow.search_runs(experiment_ids=[experiment_id], order_by=["start_time DESC"], max_results=50)

    run_name_map = {
        "da83cb432e654ccbaf580cb3ad5e8a53": "Classification - XGBoost",
        "ba71f4cd93be480ab1ed7365eab3bbcf": "Classification - Random Forest",
        "06a9695fd2ab4bcc8b57b4791db89cd5": "Classification - Logistic Regression",
        "4b2005a1108f4a63b2e4ecf45c550b20": "Regression - XGBoost Regressor",
        "b2e0c8de8df248f987c9cf8e5d9e0f03": "Regression - Random Forest Regressor",
        "ff87ae9d692a4645a00a779d97b8db2b": "Regression - Linear Regression"
    }

    runs_df["manual_name"] = runs_df["run_id"].map(run_name_map).fillna("Unnamed")
    runs_df["label"] = runs_df.apply(lambda row: f"{row['manual_name']} ({row['run_id'][:8]})", axis=1)
    label_to_id = dict(zip(runs_df["label"], runs_df["run_id"]))
    selected_labels = st.multiselect("Select up to 3 models to compare", options=list(label_to_id.keys()), max_selections=3)
    selected_ids = [label_to_id[label] for label in selected_labels]

    st.markdown("""---""")
    st.markdown("""
    🔗 **Access the [MLflow UI](http://localhost:5000)** to:
    - Compare model performance
    - View metrics like F1-score, RMSE, ROC-AUC
    - Explore logged artifacts and parameters
    - Register and version models
    """)

    if selected_ids:
        compare_df = runs_df[runs_df["run_id"].isin(selected_ids)].copy()
        compare_df["start_time"] = pd.to_datetime(compare_df["start_time"])
        compare_df["end_time"] = pd.to_datetime(compare_df["end_time"])
        compare_df["duration"] = (compare_df["end_time"] - compare_df["start_time"]).dt.total_seconds().round(2)

        param_cols = [col for col in compare_df.columns if col.startswith("params.")]
        metric_cols = [col for col in compare_df.columns if col.startswith("metrics.")]

        display_df = compare_df[[
            "manual_name", "run_id", "start_time", "end_time", "duration"
        ] + param_cols + metric_cols].rename(columns={
            "manual_name": "Model Name",
            "run_id": "Run ID",
            "start_time": "Start Time",
            "end_time": "End Time",
            "duration": "Duration (s)"
        })

        st.subheader("📊 Full Run Comparison Table")
        st.dataframe(display_df.set_index("Model Name").style.format(precision=4))

        st.subheader("📈 Metric Comparison Charts")
        for metric in metric_cols:
            chart_df = compare_df[["manual_name", metric]].dropna()
            chart_df = chart_df.rename(columns={"manual_name": "Model", metric: "Value"})
            st.markdown(f"**{metric.replace('metrics.', '').upper()}**")
            st.bar_chart(chart_df.set_index("Model"))

        st.subheader("🖼️ Artifact Previews")
        for run_id in selected_ids:
            artifact_path = f"mlruns/{experiment_id}/{run_id}/artifacts"
            model_name = run_name_map.get(run_id, "Unnamed")

            st.markdown(f"**🔍 {model_name} ({run_id[:8]})**")
            if os.path.exists(f"{artifact_path}/confusion_matrix.png"):
                st.image(Image.open(f"{artifact_path}/confusion_matrix.png"), caption="Confusion Matrix", use_column_width=True)
            if os.path.exists(f"{artifact_path}/emi_scatter.png"):
                st.image(Image.open(f"{artifact_path}/emi_scatter.png"), caption="EMI Prediction Scatter Plot", use_column_width=True)
            if os.path.exists(f"{artifact_path}/credit_score_hist.png"):
                st.image(Image.open(f"{artifact_path}/credit_score_hist.png"), caption="Credit Score Distribution", use_column_width=True)
    else:
        st.info("Select up to 3 models from the dropdown above to compare their metadata, metrics, and artifacts.")


# 🛠 Tab 6:  Admin Panel
with tab6: 
    st.title("🛠 Admin Panel")

    # 📋 Show current dataset
    st.subheader("📋 Current Dataset")
    st.dataframe(df)

    # ✏️ Modify Existing Row
    st.subheader("✏️ Modify Existing Row")
    row_indices = df.index.tolist()
    selected_index = st.selectbox("Select Row Index to Modify", row_indices)

    selected_row = df.loc[selected_index]
    updated_values = {}

    with st.form("modify_form"):
        for col in df.columns:
            updated_values[col] = st.text_input(f"{col}", value=str(selected_row[col]))
        submitted = st.form_submit_button("Update Row")
        if submitted:
            for col in df.columns:
                df.at[selected_index, col] = updated_values[col]
            df.to_csv("emi_model_data.csv", index=False)
            st.success(f"✅ Row {selected_index} updated successfully!")

    # 🗑️ Delete Row
    st.subheader("🗑️ Delete Row")
    delete_index = st.selectbox("Select Row Index to Delete", row_indices, key="delete")
    if st.button("Delete Selected Row"):
        df = df.drop(index=delete_index).reset_index(drop=True)
        df.to_csv("emi_model_data.csv", index=False)
        st.success(f"🗑️ Row {delete_index} deleted successfully!")

    # ➕ Add New Row
    st.subheader("➕ Add New Row")
    new_values = {}
    with st.form("add_form"):
        for col in df.columns:
            new_values[col] = st.text_input(f"{col}", key=f"add_{col}")
        add_submitted = st.form_submit_button("Add Row")
        if add_submitted:
            df = pd.concat([df, pd.DataFrame([new_values])], ignore_index=True)
            df.to_csv("emi_model_data.csv", index=False)
            st.success("✅ New row added successfully!")

    # 📤 Upload New Dataset
    st.subheader("📤 Upload New Dataset")
    uploaded_file = st.file_uploader("Upload CSV", type="csv")
    if uploaded_file:
        new_df = pd.read_csv(uploaded_file)
        new_df.to_csv("emi_model_data.csv", index=False)
        st.success("✅ Dataset updated successfully!")

    # 📥 Download Current Dataset
    st.subheader("📥 Download Current Dataset")
    st.download_button("Download CSV", df.to_csv(index=False), file_name="emi_model_data.csv")