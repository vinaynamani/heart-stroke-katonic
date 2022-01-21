import warnings
from audioop import avg
import pickle
from re import A
import streamlit as st
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt
import requests
from PIL import Image
from io import BytesIO
import seaborn as sns
from pickle import dump, load
from matplotlib.pyplot import figure
figure(figsize=(8, 4), dpi=80)

# plt.rcParams['figure.figsize'] = 11.7, 6.27

warnings.filterwarnings("ignore")
st.set_option('deprecation.showPyplotGlobalUse', False)
plt.rcParams['savefig.dpi'] = 80
plt.rcParams['figure.dpi'] = 40
response = requests.get(url='https://katonic.ai/favicon.ico')
im = Image.open(BytesIO(response.content))

st.set_page_config(page_title='Heart-Stroke Prediction',
                   page_icon=im,
                   layout='wide',
                   initial_sidebar_state='auto')

st.sidebar.image("logo.png")
st.sidebar.title("Heart-Stroke Prediction")
st.sidebar.write("----")
st.sidebar.info(""" Heart disease, stroke, and other cardiovascular diseases cause 1 in 3 deaths in the United States.

These diseases cost the US health care system $214 billion a year and cause $138 billion in lost productivity from premature death alone.

High blood pressure, high LDL (bad) cholesterol, diabetes, and smoking are key risk factors for heart disease and stroke. """)

st.sidebar.warning("Made with :heart: by Vinay Namani")

git_url = "https://github.com/vinaynamani/heart-stroke-katonic"
str_text = "Git:Repo"
link = f'[{str_text}]({git_url})'
st.sidebar.markdown(link, unsafe_allow_html=True)

st.write("""
# Heart Stroke Prediction
This app predicts whether the patient has a chance to get Heart-Stroke or Not.
""")
st.write('---')

image = Image.open("heart.jpg")
st.image(image, use_column_width=None)

df = pd.read_csv('train.csv')

st.header("Data Preview")
preview = st.radio("Choose Head/Tail?", ('Head', 'Tail'))
if(preview == "Head"):
    st.write(df.head())
if(preview == "Tail"):
    st.write(df.tail())

if(st.checkbox("Show complete Dataset")):
    st.write(df)

if(st.checkbox("Display the shape of the Dataset")):
    st.write(df.shape)
    dim = st.radio("Choose Dimension", ('Rows', 'Columns'))
    if(dim == "Rows"):
        st.write(df.shape[0])
    if(dim == "Columns"):
        st.write(df.shape[1])
st.header("Data Visualization")
st.subheader("Seaborn - Heart Stroke Count")
st.write(sns.countplot(x='stroke', data=df))
st.pyplot()

st.subheader("Gender wise Stroke Rate")

st.write(sns.countplot(df['stroke'], hue=df['gender']))
st.pyplot()

st.subheader("Marriage wise Stroke Rate")
st.write(sns.countplot(df['stroke'], hue=df['ever_married'], palette="magma"))
st.pyplot()

st.subheader("Work Type wise Stroke Rate")
st.write(sns.countplot(df['stroke'], hue=df['work_type'], palette="plasma"))
st.pyplot()

st.subheader("Residence wise Stroke Rate")
st.write(sns.countplot(df['stroke'],
                       hue=df['Residence_type'], palette="Dark2"))
st.pyplot()

st.subheader("Smoking Status wise Stroke Rate")
st.write(sns.countplot(df['stroke'],
                       hue=df['smoking_status'], palette="magma"))
st.pyplot()

st.title("Let's predict whether you had a risk of Getting Heart Stroke")


gender = st.radio("Choose your gender", ("Male", "Female"))

age = st.slider("Choose your age", 1, 120, 1)

hypertension = st.radio("Hypertension", ("Yes", "No"))
if hypertension == "Yes":
    hypertension = 1
else:
    hypertension = 0

heartdisease = st.radio("Heart related problems", ("Yes", "No"))
if heartdisease == "Yes":
    heartdisease = 1
else:
    heartdisease = 0

ever_married = st.radio("Ever Married", ("Yes", "No"))

work_type = st.radio("Work Type", ('Private', 'Self-employed',
                                   'Govt_job', 'children', 'Never_worked'))

residence_type = st.radio("Residence Type", ("Urban", "Rural"))

avg_glucose_level = st.slider("Average Glucose Level", 50, 300, 1)


bmi = st.slider("BMI(Body Mass Index)", 1, 150, 1)

smoking_status = st.radio(
    "Smoking Status", ('formerly smoked', 'never smoked', 'smokes', 'Unknown'))

cols_to_encode = [[gender, ever_married,
                   work_type, residence_type, smoking_status]]
encoder = load(open("OneHotEncoder.pkl", 'rb'))
model = load(open("model.pkl", 'rb'))

enc_values = encoder.transform(cols_to_encode).toarray().tolist()

values = [age, hypertension, heartdisease, avg_glucose_level, bmi]
enc_values = enc_values[0]
# values = values.extend(enc_values)
values.extend(enc_values)


if st.button("Predict"):

    prediction = model.predict([values])
    if prediction == 1:
        st.warning("You highly have a chance of getting Heart Stroke")
        positive = Image.open("positive.png")
        st.image(positive, use_column_width=None)
    else:
        st.success("Thank God , You're safe from Heart Stroke")
        negative = Image.open("negative.png")
        st.image(negative, use_column_width=None)
