import streamlit as st
import numpy as np
import pickle

# Load trained model
model = pickle.load(open("model.pkl", "rb"))

st.title("🩺 Diabetes Prediction & Lifestyle recommendation")

st.write("Enter patient details below:")

# Input fields
gender = st.selectbox("Gender", ["Female", "Male"])
age = st.number_input("Age", min_value=1, max_value=120, value=25)
hypertension = st.selectbox("Hypertension", [0, 1])
heart_disease = st.selectbox("Heart Disease", [0, 1])
smoking_history = st.selectbox(
    "Smoking History",
    ["never", "No Info", "current", "former", "ever", "not current"]
)
bmi = st.number_input("BMI", min_value=10.0, max_value=60.0, value=25.0)
hba1c = st.number_input("HbA1c Level", min_value=3.0, max_value=15.0, value=5.0)
glucose = st.number_input("Blood Glucose Level", min_value=50, max_value=300, value=120)

# Encode inputs manually (must match training)
gender_val = 1 if gender == "Male" else 0

smoking_map = {
    "never": 0,
    "No Info": 1,
    "current": 2,
    "former": 3,
    "ever": 4,
    "not current": 5
}
smoking_val = smoking_map[smoking_history]

# Prediction
if st.button("Predict"):
    input_data = np.array([[gender_val, age, hypertension, heart_disease,
                            smoking_val, bmi, hba1c, glucose]])

    prediction = model.predict(input_data)[0]

    # Display risk
    if prediction == 1:
        st.error("⚠️ High risk of Diabetes")
       
        # Determine probable Type
        if age < 30 and bmi < 25:
            diabetes_type = "Type 1 (likely)"
        else:
            diabetes_type = "Type 2 (likely)"
        st.info(f"Probable Diabetes Type: {diabetes_type}")

        # Check reversibility (only for Type 2)
        if diabetes_type.startswith("Type 2"):
            if bmi < 30 and hba1c < 7.5:
                reversibility = "Possible with lifestyle changes ✅"
            else:
                reversibility = "Requires medical management ⚠️"
            st.info(f"Reversibility: {reversibility}")

        # Food & Lifestyle Recommendations
        st.subheader("🍏 Suggested Diet & Lifestyle")
        if diabetes_type.startswith("Type 2"):
            st.markdown("""
            **Recommended Foods:**  
            - Whole grains  
            - Leafy vegetables  
            - Low-GI fruits  
            - Lean proteins  

            **Foods to Avoid:**  
            - Sugary drinks  
            - White bread & refined carbs  
            - Fried foods  
            - Processed snacks  

            **Lifestyle Suggestions:**  
            - 30 minutes walking daily  
            - Weight management  
            - Stress control  
            - Regular glucose monitoring  
            """)
        else:
            st.markdown("""
            **Type 1 diabetes requires medical supervision.**  
            - Maintain regular insulin therapy  
            - Monitor glucose frequently  
            - Follow doctor-recommended diet
            """)
    else:
        st.success("✅ Low risk of Diabetes")
        st.info("Maintain a healthy lifestyle to prevent diabetes.")
