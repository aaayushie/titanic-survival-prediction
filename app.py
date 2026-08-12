import streamlit as st
import joblib
import pandas as pd

# loads the trained model we created earlier
model = joblib.load("model.pkl")

st.sidebar.title("About the Model")

st.sidebar.write("Model: Logistic Regression")
st.sidebar.write("Dataset: Titanic")
st.sidebar.write("Accuracy: 81.01%")
st.sidebar.write("Features: 7")

st.title("Titanic Survival Prediction")
st.markdown("Predict whether a Titanic passenger would have survived based on passenger details.")

col1, col2 = st.columns(2)

with col1:
    pclass = st.selectbox(
        "Passenger Class",
        [1, 2, 3]
    )

with col2:
    sex = st.selectbox(
        "Sex",
        ["Female", "Male"]
    )

with col1:
    age = st.number_input(
        "Enter your age: ",
        min_value=0,
        max_value=100,
        value=25
    )

# number of siblings/spouses aboard
with col2:
    sibsp = st.number_input(
        "Number of Siblings/Spouses",
        min_value=0,
        max_value=8,
        value=0
    )

# number of parents/children aboard
with col1:
    parch = st.number_input(
        "Number of Parents/Children",
        min_value=0,
        max_value=6,
        value=0
    )

with col2:
    fare = st.number_input(
        "Ticket Fare",
        min_value=0.0,
        value=32.0
    )

# Embarked: tells us the port where the passenger boarded
with col1:
    embarked = st.selectbox(
        "Port of embarkation",
        ["C","Q","S"]
    )

if st.button("Predict Survival"):
    input_data = pd.DataFrame({
        "Pclass": [pclass],
        "Sex": [sex.lower()],
        "Age": [age],
        "SibSp": [sibsp],
        "Parch": [parch],
        "Fare": [fare],
        "Embarked": [embarked]
    })

    prediction = model.predict(input_data)
    probability = model.predict_proba(input_data)

    survival_probability = probability[0][1]*100
    st.metric(
        "Survival Probability",
        f"{survival_probability: .2f}%"
    )
    
    if prediction[0] == 1:
        st.success("The passenger is predicted to survive!")
    else:
        st.error("The passenger is predicted not to survived.")

    