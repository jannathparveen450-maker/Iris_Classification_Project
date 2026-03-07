import pandas as pd
import streamlit as st
from sklearn import datasets
from sklearn.ensemble import RandomForestClassifier

st.title('Iris Prediction App')
st.markdown("""
# *This app predicts the Iris flower type*"""
)

st.sidebar.header('User Input Parameters')

def user_InputFeatures():
    sepal_length = st.sidebar.slider('Sepal Length', 4.2, 7.7, 5.2)
    sepal_width = st.sidebar.slider('Sepal Width', 2.0, 4.4, 3.4)
    petal_length = st.sidebar.slider('Petal Length', 1.0, 6.9, 1.5)
    petal_width = st.sidebar.slider('Petal Width', 0.1, 2.5,0.3)

    data = {
       "sepal_length":sepal_length,
       "sepal_width":sepal_width,
       "petal_length":petal_length,
       "petal_width":petal_width
    }
    features = pd.DataFrame(data,index=[0])
    return features

df = user_InputFeatures()

st.subheader("> User Input Parameters")
st.write(df)

iris = datasets.load_iris()
X = iris.data
y = iris.target

clf = RandomForestClassifier()
clf.fit(X, y)

prediction = clf.predict(df)
prediction_prob = clf.predict_proba(df)

st.subheader("""
> Class Labels and their Corresponding Index"""
)
st.write(iris.target_names)
st.subheader("> Prediction")
st.write(iris.target_names[prediction])
# st.write(prediction)

st.subheader("Prediction Probability")
st.write(prediction_prob)

st.write("""

# See You in the Next Project!:smiley:""")