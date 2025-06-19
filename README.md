👨‍💼 Employee Attrition Analysis and Prediction

This project helps analyze why employees leave a company and predicts if a current employee is likely to leave, using data analysis and machine learning. A simple web app is built using Streamlit.

🛠 Tools & Technologies

Python

Pandas, NumPy – for data handling

Matplotlib, Seaborn – for visualization

Scikit-learn – for machine learning

Streamlit – for the web dashboard

🔧 What This Project Does

Analyzes employee data to find patterns related to attrition.

Trains a machine learning model to predict attrition.

Provides an interactive web app where you can input employee details and get a prediction.

📁 Project Files

bash

Copy

Edit

employee-attrition/

├── app.py                # Streamlit app

├── employee_attrition.csv  # Dataset

├── model.pkl             # Trained model

├── scaler.pkl            # Scaler for input features

├── requirements.txt      # Required libraries

└── README.md             # Project guide

📊 About the Dataset

Contains info about employees like age, income, job role, working years, etc.

The target is Attrition – whether the employee left or stayed.

🧠 Model Info

Trained using Logistic Regression or Random Forest

Uses StandardScaler to normalize inputs

Outputs a prediction: Yes (will leave) or No (will stay)

📌 Features of the App

📉 EDA Page: Visual charts to explore the data

🧾 Prediction Page: Form to input new employee data and get prediction

📤 Real-time Output: Get instant results with probabilities

