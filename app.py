import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import pickle
import mlflow
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image

# ✅ Set full-width layout and page title
st.set_page_config(page_title="EMI Prediction AI", layout="wide")

# ✅ Centered App Title
st.markdown("<h1 style='text-align: center;'>  📊 EMI Prediction AI</h1>", unsafe_allow_html=True)

# ✅ Load trained XGBoost models
xgb_classifier = joblib.load("xgb_classifier.pkl")
xgb_regressor = joblib.load("xgb_regressor.pkl")

# ✅ Load dataset with fixes
df = pd.read_csv("emi_prediction_dataset.csv.gz", compression="gzip", low_memory=False)
df["age"] = pd.to_numeric(df["age"], errors="coerce")
df["monthly_salary"] = pd.to_numeric(df["monthly_salary"], errors="coerce")
df["credit_score"] = pd.to_numeric(df["credit_score"], errors="coerce")
df["current_emi_amount"] = pd.to_numeric(df.get("current_emi_amount", pd.Series()), errors="coerce")

# ✅ Layout: padded columns with centered tab area
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

# 🏠 Tab 1: Home
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
    sns.histplot(df["credit_score"].dropna(), bins=30, ax=ax)
    st.pyplot(fig)

    st.subheader("Correlation Heatmap")
    fig, ax = plt.subplots()
    sns.heatmap(df.corr(numeric_only=True), cmap="coolwarm", ax=ax)
    st.pyplot(fig)

    st.subheader("Monthly Salary Distribution")
    fig, ax = plt.subplots()
    sns.histplot(df["monthly_salary"].dropna(), bins=30, kde=True, color="skyblue", ax=ax)
    st.pyplot(fig)

    st.subheader("Max EMI vs Credit Score")
    fig, ax = plt.subplots()
    if "max_monthly_emi" in df.columns:
        sns.scatterplot(data=df, x="credit_score", y="max_monthly_emi", hue="gender", ax=ax)
        st.pyplot(fig)

    st.subheader("Max EMI vs Requested Tenure")
    fig, ax = plt.subplots()
    if "max_monthly_emi" in df.columns:
        sns.lineplot(data=df, x="requested_tenure", y="max_monthly_emi", ax=ax)
        st.pyplot(fig)

    st.subheader("Applicant Count by Gender")
    fig, ax = plt.subplots()
    sns.countplot(data=df, x="gender", hue="gender", palette="pastel", ax=ax, legend=False)
    st.pyplot(fig)

    st.subheader("Current EMI by Existing Loans")
    fig, ax = plt.subplots()
    if "current_emi_amount" in df.columns:
        sns.boxplot(data=df, x="existing_loans", y="current_emi_amount", hue="existing_loans", palette="Set2", ax=ax, legend=False)
        st.pyplot(fig)
#✅ Tab 3: EMI Eligibility Predictor
with tab3:
    st.title("✅ EMI Eligibility Predictor")

    gender = st.selectbox("Gender", ["Male", "Female"])
    employment_type = st.selectbox("Employment Type", ["Salaried", "Self-Employed", "Freelancer"])
    monthly_salary = st.number_input("Monthly Salary", min_value=0)
    credit_score = st.slider("Credit Score", min_value=300, max_value=900, value=750)
    existing_loans = st.selectbox("Existing Loans", ["No", "Yes"])
    monthly_rent = st.number_input("Monthly Rent", min_value=0)
    school_fees = st.number_input("School Fees", min_value=0)
    college_fees = st.number_input("College Fees", min_value=0)
    dependents = st.number_input("Number of Dependents", min_value=0)
    other_expenses = st.number_input("Other Monthly Expenses", min_value=0)
    requested_amount = st.number_input("Requested Loan Amount", min_value=0)
    requested_tenure = st.slider("Requested Tenure (months)", min_value=1, max_value=120, value=12)

    if st.button("Predict Eligibility"):
        # ✅ Rule-based override
        if monthly_salary < 5000 and requested_amount > 10 * monthly_salary:
            st.error("❌ Not Eligible (Rule-based override: salary too low for requested amount)")
        else:
            input_dict = {
                "gender": 0 if gender == "Male" else 1,
                "monthly_salary": monthly_salary,
                "credit_score": credit_score,
                "existing_loans": 0 if existing_loans == "No" else 1,
                "monthly_rent": monthly_rent,
                "school_fees": school_fees,
                "college_fees": college_fees,
                "dependents": dependents,
                "other_monthly_expenses": other_expenses,
                "requested_amount": requested_amount,
                "requested_tenure": requested_tenure,
                "employment_type": {"Salaried": 0, "Self-Employed": 1, "Freelancer": 2}[employment_type]
            }

            input_df = pd.DataFrame([input_dict])
            try:
                prediction = xgb_classifier.predict(input_df)[0]
                prob = xgb_classifier.predict_proba(input_df)[0][prediction]
                label_map = {0: "Not Eligible", 1: "High Risk", 2: "Eligible"}
                label = label_map.get(prediction, "Unknown")

                if prediction == 2:
                    st.success(f"✅ {label} (Confidence: {prob:.2f})")
                elif prediction == 1:
                    st.warning(f"⚠️ {label} (Confidence: {prob:.2f})")
                else:
                    st.error(f"❌ {label} (Confidence: {prob:.2f})")
            except Exception as e:
                st.warning(f"⚠️ Prediction failed: {e}")


# 💰 Tab 4: EMI Amount Predictor
with tab4:
    st.title("💰 EMI Amount Predictor")

    gender = st.selectbox("Gender", ["Male", "Female"], key="emi_gender")
    employment_type = st.selectbox("Employment Type", ["Salaried", "Self-Employed", "Freelancer"], key="emi_employment")
    monthly_salary = st.number_input("Monthly Salary", min_value=0, key="emi_salary")
    credit_score = st.slider("Credit Score", min_value=300, max_value=900, value=750, key="emi_credit")
    existing_loans = st.selectbox("Existing Loans", ["No", "Yes"], key="emi_loans")
    monthly_rent = st.number_input("Monthly Rent", min_value=0, key="emi_rent")
    school_fees = st.number_input("School Fees", min_value=0, key="emi_school")
    college_fees = st.number_input("College Fees", min_value=0, key="emi_college")
    dependents = st.number_input("Number of Dependents", min_value=0, key="emi_dependents")
    other_expenses = st.number_input("Other Monthly Expenses", min_value=0, key="emi_other")
    requested_amount = st.number_input("Requested Loan Amount", min_value=0, key="emi_amount")
    requested_tenure = st.slider("Requested Tenure (months)", min_value=1, max_value=120, value=12, key="emi_tenure")

    if st.button("Predict EMI Amount"):
        # ✅ Rule-based override
        if monthly_salary < 5000 and requested_amount > 10 * monthly_salary:
            st.error("❌ EMI prediction blocked (Rule-based override: salary too low for requested amount)")
        else:
            input_dict = {
                "gender": 0 if gender == "Male" else 1,
                "monthly_salary": monthly_salary,
                "credit_score": credit_score,
                "existing_loans": 0 if existing_loans == "No" else 1,
                "monthly_rent": monthly_rent,
                "school_fees": school_fees,
                "college_fees": college_fees,
                "dependents": dependents,
                "other_monthly_expenses": other_expenses,
                "requested_amount": requested_amount,
                "requested_tenure": requested_tenure,
                "employment_type": {"Salaried": 0, "Self-Employed": 1, "Freelancer": 2}[employment_type]
            }

            input_df = pd.DataFrame([input_dict])
            try:
                emi_prediction = xgb_regressor.predict(input_df)[0]
                st.success(f"💰 Predicted EMI Amount: ₹{emi_prediction:,.2f}")
            except Exception as e:
                st.warning(f"⚠️ Prediction failed: {e}")


# Tab 5: 📈 Model Monitoring
with tab5:
    st.title("📈 Model Monitoring")

    import os
    import pandas as pd
    from PIL import Image

    def is_running_locally():
        return os.getenv("STREAMLIT_SERVER_HOST") is None

    if is_running_locally():
        try:
            import mlflow

            mlflow.set_tracking_uri("http://127.0.0.1:5000")
            experiment = mlflow.get_experiment_by_name("EMI_Prediction_Experiment")

            if experiment is not None:
                experiment_id = experiment.experiment_id

                runs_df = mlflow.search_runs(
                    experiment_ids=[experiment_id],
                    order_by=["start_time DESC"],
                    max_results=50
                )

                run_name_map = {
                    "2aa683bffcfd4f3fa55e7b3ecb01564c": "Classification - XGBoost",
                    "e09fad24aa49484eace6c780edef15f7": "Classification - Random Forest",
                    "a7ccd517a4704b27b0261665c703fe9a": "Classification - Logistic Regression",
                    "34daf67c53b648409a84992dcf7c1031": "Regression - XGBoost Regressor",
                    "3e8f66c20e1047b3958f055963c8a3b9": "Regression - Random Forest Regressor",
                    "04057cf9c4a74bfdbd36e5e0b586b3ed": "Regression - Linear Regression"
                }

                runs_df["manual_name"] = runs_df.apply(
                    lambda row: row.get("tags.model_name") or run_name_map.get(row["run_id"], "Unnamed"),
                    axis=1
                )
                runs_df["label"] = runs_df.apply(lambda row: f"{row['manual_name']} ({row['run_id'][:8]})", axis=1)
                label_to_id = dict(zip(runs_df["label"], runs_df["run_id"]))
                selected_labels = st.multiselect(
                    "Select up to 3 models to compare",
                    options=list(label_to_id.keys()),
                    max_selections=3
                )
                selected_ids = [label_to_id[label] for label in selected_labels]

                st.markdown("---")
                st.markdown("""
                🔗 **Access the [MLflow UI](http://127.0.0.1:5000)** to:
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
                        model_name = runs_df.loc[runs_df["run_id"] == run_id, "manual_name"].values[0]

                        st.markdown(f"**🔍 {model_name} ({run_id[:8]})**")
                        st.markdown(f"🔗 [View Run in MLflow UI](http://127.0.0.1:5000/#/experiments/{experiment_id}/runs/{run_id})")

                        for filename, caption in {
                            "confusion_matrix.png": "Confusion Matrix",
                            "emi_scatter.png": "EMI Prediction Scatter Plot",
                            "credit_score_hist.png": "Credit Score Distribution"
                        }.items():
                            file_path = os.path.join(artifact_path, filename)
                            if os.path.exists(file_path):
                                st.image(Image.open(file_path), caption=caption, use_column_width=True)
                else:
                    st.info("Select up to 3 models from the dropdown above to compare their metadata, metrics, and artifacts.")
            else:
                st.warning("⚠️ MLflow experiment 'EMI_Prediction_Experiment' not found.")
        except Exception:
            st.info("⚠️ MLflow tracking is disabled in cloud mode. To view model metrics and artifacts, please run this dashboard locally with MLflow server active.\n\n"
            "🔗 [Open MLflow UI locally](http://127.0.0.1:5000)")
        
        # 🌐 Cloud fallback message
        st.info(
            "⚠️ MLflow tracking is disabled in cloud mode. To view model metrics and artifacts, please run this dashboard locally with MLflow server active.\n\n"
            "🔗 [Open MLflow UI locally](http://127.0.0.1:5000)"
        )



# 🛠 Tab 6: Admin Panel
with tab6:
    st.title("🛠 Admin Panel")

    # ✅ Define dataset path
    DATA_PATH = "emi_model_data.csv.zip"

    # ✅ Load dataset into session state
    if "df" not in st.session_state:
        st.session_state.df = pd.read_csv(DATA_PATH)

    df = st.session_state.df

    # 📋 Show current dataset
    st.subheader("📋 Current Dataset")
    st.dataframe(df)

    # 🔄 Refresh Dataset
    if st.button("🔄 Refresh Dataset"):
        st.session_state.df = pd.read_csv(DATA_PATH)
        df = st.session_state.df
        st.success("✅ Dataset reloaded from file.")

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
            df.to_csv(DATA_PATH, index=False)
            st.session_state.df = df
            st.success(f"✅ Row {selected_index} updated successfully!")

    # 🗑️ Delete Row
    st.subheader("🗑️ Delete Row")
    delete_index = st.selectbox("Select Row Index to Delete", row_indices, key="delete")
    if st.button("Delete Selected Row"):
        df = df.drop(index=delete_index).reset_index(drop=True)
        df.to_csv(DATA_PATH, index=False)
        st.session_state.df = df
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
            df.to_csv(DATA_PATH, index=False)
            st.session_state.df = df
            st.success("✅ New row added successfully!")

    # 📤 Upload New Dataset
    st.subheader("📤 Upload New Dataset")
    uploaded_file = st.file_uploader("Upload CSV", type="csv")
    if uploaded_file:
        new_df = pd.read_csv(uploaded_file)
        new_df.to_csv(DATA_PATH, index=False)
        st.session_state.df = new_df
        st.success("✅ Dataset updated successfully!")

    # 📥 Download Current Dataset
    st.subheader("📥 Download Current Dataset")
    st.download_button("Download CSV", df.to_csv(index=False), file_name="emi_model_data.csv")