import streamlit as st
import numpy as np
import joblib
import sklearn.metrics as sm
st.set_page_config(page_title="Calorie Prediction App")
d1={'Gender': 2.2357162090988796,
 'Age': 15.43951313410011,
 'Height': 1.7536767874796058,
 'Weight': 3.548058186832146,
 'Duration': 95.54205334742683,
 'Heart_Rate': 89.7882060638828,
 'Body_Temp': 82.45577577192238}
test=np.load('Testdata.npy')
knn=np.load('KNNPred.npy')
rf=np.load('RandomForest.npy')
svm=np.load('SVM.npy')
st.title("**_BURNED CALORIE PREDICTION_**")
Gender=st.selectbox('select Gender',('male','female'))
if Gender=='male':
    Gender=1
elif Gender=='female':
    Gender=0
Age=st.text_input('Age')
Height=st.text_input('Height (in centimeters)')
Weight=st.text_input('Weight')
Duration=st.text_input('Duration')
HeartRate=st.text_input('Heart Rate')
BodyTemperature=st.text_input('Body Temp')
option=st.sidebar.selectbox('select the model',('RandomForest','SVM','KNN','Accuracy'))
def BMI():
    if bmi<18.5:
        return ' (Under Weight)'
    elif 18.5<bmi<24.9:
        return ' (Normal)'
    elif 25<=bmi<=29.9:
        return ' (Over Weight)'
    elif 30<=bmi<=39.9:
        return ' (Obesity)'
    else:
        return' (Extreme obesity)'
if option=='Accuracy':
    a1=str(sm.r2_score(test,knn)*100)
    a2=str(sm.r2_score(test,rf)*100)
    a=str(sm.r2_score(test,svm)*100)
    st.sidebar.header('KNN:'+a1+'%')
    st.sidebar.header('RandomForest:'+a2+'%')
    st.sidebar.header('SVM:'+a+'%')
if st.button('predict'):
    if option=='Accuracy':
        st.error("select model for prediction(not Accuracy)")
    elif option=='Dependency':
        st.error("select model for prediction(not Dependency)")
    elif any(i=='' for i in [Age,Height,Weight,Duration,HeartRate,BodyTemperature]):
        st.warning("enter all values")
    else:
        if option=='RandomForest':
            model=joblib.load('RandomForest')
            op=model.predict([[Gender,Age,Height,Weight,Duration,HeartRate,BodyTemperature]])
        elif option=='SVM':
            model=joblib.load('Svm')
            Age=int(Age)
            Height=int(Height)
            Weight=int(Weight)
            Duration=int(Duration)
            HeartRate=int(HeartRate)
            BodyTemperature=int(BodyTemperature)
            op=model.predict([[Gender,Age,Height,Weight,Duration,HeartRate,BodyTemperature]])
        elif option=='KNN':
            model=joblib.load('KNN')
            Age=int(Age)
            Height=int(Height)
            Weight=int(Weight)
            Duration=int(Duration)
            HeartRate=int(HeartRate)
            BodyTemperature=int(BodyTemperature)
            op=model.predict([[Gender,Age,Height,Weight,Duration,HeartRate,BodyTemperature]])
        cal=str(int(op[0]))
        st.title(' calories: '+cal)
        bmi=int(Weight)/((int(Height)/100)**2)
        st.header('BMI: '+str(bmi)+BMI())
        st.snow()
