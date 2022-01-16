#Import libraries
import streamlit as st
import pandas as pd
import numpy as np
# import matplotlib.pyplot as plt
from PIL import Image


from sklearn.preprocessing import LabelEncoder,MinMaxScaler 
le = LabelEncoder()
scaler = MinMaxScaler(feature_range=(0, 1))

#Saving best model 
import joblib

import warnings
warnings.filterwarnings('ignore')


#load the model from models dir

model2 = joblib.load(r"model.pkl")

#Import python scripts
from preprocessing import preprocess_dropout


def main():
    #Setting Application title
    st.title('Digital Learning & Education App')

      #Setting Application description
    st.markdown("""
     :dart:  This Streamlit app is made to predict the performance of a students and likehood of a student to dropout from school due to diffrent factor
    The application is functional for both online prediction and batch data prediction. \n
    """)
    st.markdown("<h3></h3>", unsafe_allow_html=True)

     #Setting Application sidebar default
    image = Image.open('download.jpeg')
    page = st.sidebar.selectbox("Dropout prediction", ['Dropout Prediction'])

    st.sidebar.image(image)

    st.title('Student Dropout Prediction')
    st.text('Select a page in the sidebar')

    st.info("Input data below")
            #Based on our optimal features selection
    st.subheader("Student informations:")
    sem1kt = st.number_input('Semester 1 KT ? ',min_value=0, max_value=5, value=0)
    sem2sgpa = st.number_input('Semester  2 SGPA ?',min_value=0, max_value=5, value=0)
    sem2kt = st.number_input('Semester 2 KT ?',min_value=0, max_value=5, value=0)
    sem4sgpa = st.number_input('Semester 4 SGPA ? ',min_value=0, max_value=5, value=0)
    sem5sgpa = st.number_input('Semester  5 SGPA ?',min_value=0, max_value=5, value=0)
    sem6sgpa = st.number_input('Semester 6 SGPA ?',min_value=0, max_value=5, value=0)
    sem7sgpa = st.number_input('Semester 7 SGPA ?',min_value=0, max_value=5, value=0)
    sem8sgpa = st.number_input('Semester  8 SGPA ?',min_value=0, max_value=5, value=0)
    Hour_per_week_wriassignment = st.selectbox('How many hours per week do you spend on writing assignments?', ('1 - 5 hours', '5 - 10 hours', 'More than 10 hours','Less than 1 hour'))
    time_to_reach_college = st.selectbox('How much time does it take for you to reach college?', ('2 - 3 hours', '1 - 2 hours', 'Less than 1 hour','More than 3 hours'))
    Averageattendence = st.selectbox('How much would you consider as your average attendance throughout all the semesters so far?', ('70%  - 79%', '80%  - 89%', '90%  - 100%', '60%  - 69%','Less than 60%'))            
    Internetathome = st.selectbox('Internet availability at home?', ('Yes', 'No'))
    hrstraightlecture = st.selectbox('Can you sit a lecture for 2 hrs straight? ', ('No', 'Yes'))
    submission_on_time = st.selectbox('Do you do your submissions on time?', ('Yes', 'No'))
    Five_lecture = st.selectbox('If there are 5 hrs of lectures per day, would you attend all?', ('Yes', 'No'))
    practical = st.selectbox('If there are 5 hrs of practicals per day, would you attend all? ', ('No', 'Yes'))
    Feedback = st.selectbox('Your teacher feedbacks', ('Good student', 'Good leadership skills', 'Hard working','Responsible', 'Disciplined and hard working',
                                     'Good but can be better', 'Excellent performance', 'willingness to put effort', 'Willingness to Put Forth Effort',
                                    'Respectful to Authority and Others','Solid Social and Emotional Skills', 'Self-Motivated',
                                     'Eagerness to Learn', 'Not attentive','Does not follow my lecture', 'Very talkative','Needs improvement', 'Poor attendance', 'Lagging Behind', 'Disappointed performance', 'Argues with teacher', 'Bunk lectuer','Always late', 'Always late and Does not follow my lecture','Does not follow my lecture and Very talkative','Very talkative and Poor attendance','Argues with teacher and Bunk lectuer','Needs improvement and Not attentive'))
    transportation = st.selectbox('What is your preferred mode of transportation to reach college?', ('Train', 'Private Vehicle', 'Bus'))
    coaching = st.selectbox('Have you enrolled for coaching classes?', ('No', 'Yes'))

    data = {
                    'SEM 1 KT':sem1kt,
                    'SEM 2 SGPA':sem2sgpa,
                    'SEM 2 KT':sem2kt,
                    'SEM 4 SGPA':sem4sgpa,
                    'SEM 5 SGPA':sem5sgpa,
                    'SEM 6 SGPA':sem6sgpa,
                    'SEM 7 SGPA':sem7sgpa,
                    'SEM 8 SGPA':sem8sgpa,
                    'Hour_per_week_wriassignment':Hour_per_week_wriassignment,
                    'time_to_reach college':time_to_reach_college, 
                    'Average attendence':Averageattendence, 
                    'Internet at home':Internetathome, 
                    '2 hr straight lecture':hrstraightlecture, 
                    'submission on time':submission_on_time, 
                    'Five lecture straight,woulf you attend all?':Five_lecture, 
                    'Five hr practical staight,do you attend all':practical,
                    'Feedback of teacher':Feedback,
                    'preffered transportatin to college':transportation, 
                    'Enrolled to coaching class':coaching, 
                    }
    features_df = pd.DataFrame.from_dict([data])
    st.markdown("<h3></h3>", unsafe_allow_html=True)
    st.write('Overview of input is shown below')
    st.markdown("<h3></h3>", unsafe_allow_html=True)
    st.dataframe(features_df)

            #Preprocess inputs
    preprocess_df = preprocess_dropout(features_df, 'Online')

    prediction = model2.predict(preprocess_df)
    if st.button('Predict'):
        if prediction == 1:
            st.warning('Dropout')
        else:
            st.success('Not Dropout')
                
if __name__ == '__main__':
        main()