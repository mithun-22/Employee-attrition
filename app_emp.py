import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

# 🌐 Set page configuration
st.set_page_config(page_title="Employee Attrition Dashboard", layout="wide")

# 🎨 Background Image & Input Styling
st.markdown(
    """
    <style>
    .stApp {
        background-image: url('https://images.unsplash.com/photo-1605902711622-cfb43c44367f?auto=format&fit=crop&w=1950&q=80');
        background-size: cover;
        background-attachment: fixed;
        background-repeat: no-repeat;
        background-position: center;
    }
    /* Make all cards and widgets semi-transparent */
    .block-container {
        background-color: rgba(255, 255, 255, 0.85);
        padding: 2rem;
        border-radius: 10px;
    }
    /* Style input boxes */
    input[type="number"], .stSlider {
        border: 2px solid #007BFF !important;
        border-radius: 5px;
        padding: 5px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# 🚀 Load data and models
@st.cache_data
def load_data():
    return pd.read_excel("Employee-Attrition.xlsx")

@st.cache_resource
def load_model_and_scaler():
    model = joblib.load("performance_rating_model.pkl")
    scaler = joblib.load("scaler.pkl")
    return model, scaler

df = load_data()
model, scaler = load_model_and_scaler()

# 🔷 Title
st.markdown("<h1 style='text-align: center; color: #003366;'>🔍 Employee Attrition Dashboard</h1>", unsafe_allow_html=True)

# 🔘 Sidebar Navigation
option = st.sidebar.radio("Navigation", ["📊 EDA", "📈 Prediction"])

# -------------------------------- EDA --------------------------------
if option == "📊 EDA":
    st.markdown("## 📊 Exploratory Data Analysis")

    st.markdown("### Dataset Preview")
    st.dataframe(df.head())

    st.markdown("### Attrition Distribution")
    attr_counts = df['Attrition'].value_counts()
    fig1, ax1 = plt.subplots()
    ax1.pie(attr_counts, labels=attr_counts.index, autopct='%1.1f%%', startangle=90, colors=['skyblue', 'salmon'])
    st.pyplot(fig1)

    st.markdown("### Correlation Heatmap")
    corr = df.select_dtypes(include=['int64', 'float64']).corr()
    fig2, ax2 = plt.subplots(figsize=(12, 6))
    sns.heatmap(corr, annot=True, cmap='coolwarm', ax=ax2)
    st.pyplot(fig2)

    st.markdown("### Feature Distribution")
    col = st.selectbox("Choose a feature", df.columns)
    if df[col].dtype == 'object':
        st.bar_chart(df[col].value_counts())
    else:
        fig3, ax3 = plt.subplots()
        sns.histplot(df[col], kde=True, ax=ax3)
        st.pyplot(fig3)

# ---------------------------- Prediction -----------------------------
else:
    st.markdown("## 🎯 Predict Employee Attrition")

    with st.form("prediction_form"):
        st.markdown("### ✏️ Enter Employee Details")
        col1, col2, col3 = st.columns(3)

        with col1:
            JobLevel = st.slider("Job Level", 1, 5, 2)
            TotalWorkingYears = st.slider("Total Working Years", 0, 40, 10)

        with col2:
            YearsInCurrentRole = st.slider("Years in Current Role", 0, 20, 5)
            YearsAtCompany = st.slider("Years at Company", 0, 40, 6)

        with col3:
            MonthlyIncome = st.number_input("Monthly Income", 1000, 50000, 5000)
            YearsWithCurrManager = st.slider("Years with Current Manager", 0, 20, 3)

        submit = st.form_submit_button("🔎 Predict")

    if submit:
        input_df = pd.DataFrame({
            'JobLevel': [JobLevel],
            'TotalWorkingYears': [TotalWorkingYears],
            'YearsInCurrentRole': [YearsInCurrentRole],
            'MonthlyIncome': [MonthlyIncome],
            'YearsAtCompany': [YearsAtCompany],
            'YearsWithCurrManager': [YearsWithCurrManager]
        })

        scaled_input = scaler.transform(input_df)
        prediction = model.predict(scaled_input)[0]
        prediction_proba = model.predict_proba(scaled_input)[0][1]

        st.markdown("### 📢 Prediction Result")
        if prediction == 1:
            st.error(f"🚨 High Risk: This employee is likely to **leave**. (Probability: {prediction_proba:.2f})")
        else:
            st.success(f"✅ Low Risk: This employee is likely to **stay**. (Probability: {prediction_proba:.2f})")
