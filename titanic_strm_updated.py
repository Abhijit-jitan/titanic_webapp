import pandas as pd
import streamlit as st
import pickle

model = pickle.load(open("titanic_v4.pkl", 'rb'))


def input_value():
    st.title("Titanic Survival Prediction")
    st.write(""" Would you have survived the Titanic Disaster?""")

    st.write("""How it's working:""")

    st.title("--- Check Your Survival Chances ---")

    age = st.slider("Enter Age :", 1, 75, 30)
    fare = st.slider("Fare (in 1912 $) :", 15, 500, 40)
    sib = st.selectbox("How many Siblings or spouses are travelling with you?", [0, 1, 2, 3, 4, 5, 6, 7, 8])
    parch = st.selectbox("How many Parents or children are travelling with you?", [0, 1, 2, 3, 4, 5, 6, 7, 8])
    sex = st.selectbox("Select Gender:", ["male", "female"])
    Sex = 1 if sex == "female" else 0
    # Sex_male = 0 if Sex == "female" else 1

    pclass = st.selectbox("Select Passenger-Class:", [1, 2, 3])
    boarding = st.selectbox("Boarded Location:", ["Cherbourg", "Queenstown", "Southampton"])
    embarked_C = 1 if boarding == "Cherbourg" else 0
    embarked_Q = 1 if boarding == "Queenstown" else 0
    embarked_S = 1 if boarding == "Southampton" else 0

    input = {"Age": age, "Fare": fare, "SibSp": sib, "Parch": parch, "Sex": Sex, "Pclass": pclass, "Embarked_C": embarked_C, "embarked_Q": embarked_Q, "Embarked_S": embarked_S}

    df = pd.DataFrame(input, index=[0])
    return df


# def predict(df):
data = input_value()

prediction = model.predict(data)
st.title("Survival Prediction")
if prediction[0] == 1:
    st.write("**You probably would have made it!**")
else:
    st.write("Well...how do I put this...**you're probably safer just watching the movie!**")
