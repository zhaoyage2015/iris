import streamlit as st
import joblib
import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt

model = joblib.load('catboost_model.pkl')


st.title("CatBoost Prediction with SHAP Explanation")


st.header("Enter input values")
sepal_length = st.number_input('Sepal length:', min_value=4.0, max_value=8.0, value=5.1)
sepal_width = st.number_input('Sepal width:', min_value=2.0, max_value=4.5, value=3.5)
petal_length = st.number_input('Petal length:', min_value=1.0, max_value=7.0, value=1.4)
petal_width = st.number_input('Petal width:', min_value=0.1, max_value=2.5, value=0.2)


feature_values = [sepal_length, sepal_width, petal_length, petal_width]
features = np.array([feature_values])

if st.button("Predict"):
    prediction = model.predict(features)[0]
    probability = model.predict_proba(features)[0]

    st.write(f"**Predicted class:** {prediction}")
    st.write(f"**Prediction probability:** {probability}")


    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(pd.DataFrame([feature_values], columns=feature_names))

    st.header("SHAP Force Plot")

    shap.force_plot(explainer.expected_value, shap_values[0], pd.DataFrame([feature_values], columns=feature_names), matplotlib=True)
    plt.savefig("shap_force_plot.png", bbox_inches='tight', dpi=1200)
    st.image("shap_force_plot.png")

