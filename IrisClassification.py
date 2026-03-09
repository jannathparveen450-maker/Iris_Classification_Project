import pandas as pd
import streamlit as st
from sklearn import datasets
from sklearn.ensemble import RandomForestClassifier

st.title("IRIS FLOWER PREDICTION APP 🌸")

st.markdown("""
             *This app predicts the Iris flower type*
            """)

st.sidebar.header("User Input Parameters")

# Function to collect user input from sidebar sliders
def user_InputFeatures():
      # Create sliders for user input
      sepal_length = st.sidebar.slider("Sepal Length",4.2,7.7,5.2)
      sepal_width = st.sidebar.slider("Sepal Width",2.0,4.4,3.4)
      petal_length = st.sidebar.slider("Petal Length",1.0,6.9,1.5)
      petal_width = st.sidebar.slider("Petal Width",0.1,2.5,0.3)

      # Store the user input values in a dictionary
      data = {
            "Sepal_length" : sepal_length,
            "Sepal_width" : sepal_width,
            "Petal_length" : petal_length,
            "Petal_width" : petal_width
      }

      # Convert dictionary to pandas DataFrame for model prediction
      features = pd.DataFrame(data, index=[0])
      return features

df = user_InputFeatures()

# Display user input parameters in the app
st.subheader(">User Input Parameters")
st.write(df)

#Load the Iris dataset
iris = datasets.load_iris()

#separate features (X) and target (y)
X = iris.data
y = iris.target

#Create Random Forest Classifier model
model = RandomForestClassifier()

#Train the model using dataset
model.fit(X,y)

#Make prediction based on user input features
prediction = model.predict(df)

#get prediction probabilities for each class
prediction_prob = model.predict_proba(df)

# Display class labels and their corresponding index
st.subheader("""
             >Class Labels and their Corresponding Index
             """)
for i , name in enumerate (iris.target_names):
      st.write(f"{i} : {name}")

#Display the predicted class
st.subheader(">Prediction")
st.write(iris.target_names[prediction])

#Display probability for each class
st.subheader(">Prediction Probability")
st.write(prediction_prob)

st.write("""
         # See You In Next Project!:smiley:
         """)



