import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')
import streamlit as st
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder, StandardScaler

data = pd.read_csv('expresso_processed.csv')

import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
import streamlit as st
from sklearn.linear_model import LogisticRegression
import joblib 


df = data.copy()

encoder1 = LabelEncoder()
encoder2 = LabelEncoder()
scaler = StandardScaler()

df.drop(['Unnamed: 0'], axis = 1, inplace = True)

for i in df.drop('CHURN', axis = 1).columns:
    if i == 'TENURE':
        df[i] = encoder1.fit_transform(df[i])
    elif i == 'MRG' :
        df[i] = encoder2.fit_transform(df[i])
    else:
        df[i] = scaler.fit_transform(df[[i]])

x = df.drop('CHURN', axis = 1)
y = df.CHURN

xtrain, xtest, ytrain, ytest = train_test_split(x,y, test_size= 0.20, stratify= y)

model = LogisticRegression()
model.fit(xtrain, ytrain)



st.markdown("<h1 style = 'color: #FFFF00; text-align: center; font-family: helvetica '>Prediction of Customer Loyalty</h1>", unsafe_allow_html =True)
st.markdown("<h6 style = 'color: #FFFF00; text-align: center; font-family: 'Arial', sans-serif '>Revolutionizing the telecom sector, our cutting-edge app leverages predictive analytics to forecast and quantify customer loyalty. With a keen focus on enhancing user experience, the application employs advanced algorithms to analyze user behavior, preferences, and engagement patterns. By providing actionable insights, telecom providers can proactively address customer needs, optimize service offerings, and foster lasting relationships, ultimately boosting customer loyalty and retention in this dynamic and competitive industry.</h6>", unsafe_allow_html =True)

st.image('pngwing.com (8).png', width = 150, use_column_width=True)
st.markdown('<br>', unsafe_allow_html = True)
st.markdown("<h4 style = 'margin: -30px; color: #1E90FF; text-align: center; font-family: sans-serif'>Designed and built by", unsafe_allow_html =True)
st.markdown('<br>', unsafe_allow_html = True)
st.markdown("<h3 style = 'margin: -30px; color: #FFA500; text-align: center; font-family: cursive '>kaka Tech Word", unsafe_allow_html =True)

st.markdown("<h4 style = 'margin: -30px; color: #FFA500; text-align: center; font-family: cursive '>Giving solution to mankind", unsafe_allow_html =True)
st.markdown('<br>', unsafe_allow_html = True)

#st.sidebar.header("DATA")
st.sidebar.markdown("""
    <div style="display: flex; justify-content: center;">
<h1>DATA</h1>
    </div>
""", unsafe_allow_html=True)
st.sidebar.image('pngwing.com (9).png')
st.sidebar.dataframe(data,use_container_width=True )
st.sidebar.markdown('<br>', unsafe_allow_html = True)

col1, col2, col3 = st.columns(3)

# Add content to each column
with col1:
    st.markdown("<h4 style='color: #1E90FF;'>Tenure</h3>",unsafe_allow_html=True)
    TENURE = st.selectbox("duration in the network", data['TENURE'].unique())
   

with col2:
    st.markdown("<h4 style='color: #1E90FF;'>Amount</h3>",unsafe_allow_html=True)
    MONTANT = st.number_input("top-up amount",data['MONTANT'].min(),data['MONTANT'].max())
with col3:
    st.markdown("<h4 style='color: #1E90FF;'>Top_up Frequency</h3>", unsafe_allow_html=True)
    FREQUENCE_RECH = st.number_input('number of times the customer refilled', data['FREQUENCE_RECH'].min())

col4, col5, col6 = st.columns(3)
with col4:
    st.markdown("<h5 style='color: #1E90FF;'>Monthly Income</h3>",unsafe_allow_html=True)
    REVENUE = st.number_input("monthly income of client")
   

with col5:
    st.markdown("<h5 style='color: #1E90FF;'>Average Qrt income</h3>",unsafe_allow_html=True)
    ARPU_SEGMENT = st.number_input("income over 90 days / 3")
with col6:
    st.markdown("<h5 style='color: #1E90FF;'>FREQUENCE</h3>", unsafe_allow_html=True)
    FREQUENCE = st.number_input('number of times the client has made an income', data['FREQUENCE'].min(), data['FREQUENCE'].max())
col7, col8, col9= st.columns(3)
with col7:
    st.markdown("<h5 style='color: #1E90FF;'>Data Volume</h3>",unsafe_allow_html=True)
    DATA_VOLUME = st.selectbox("number of connections", data['REVENUE'])
   

with col8:
    st.markdown("<h5 style='color: #1E90FF;'>ON_NET</h3>",unsafe_allow_html=True)
    ON_NET = st.number_input("calls within line")
with col9:
    st.markdown("<h5 style='color: #1E90FF;'>MRG</h3>",unsafe_allow_html=True)
    MRG = st.selectbox("a client who is going",data['MRG'].unique())
st.markdown("<h5 style='color: #1E90FF;'>REGULARITY</h3>",unsafe_allow_html=True)
REGULARITY = st.number_input("number of times the client is active for 90 days")

#TENURE = st.selectbox("duration in the network", data['TENURE'].unique())



model = joblib.load('ChunPrediction.pkl')
new_tenure = encoder1.transform([TENURE])
new_mrs = encoder2.transform([MRG])


st.write('Input Variables')
input_var = pd.DataFrame({'TENURE':[new_tenure],
                          'MONTANT':[MONTANT], 'FREQUENCE_RECH':[FREQUENCE_RECH],
                          'REVENUE':[REVENUE], 'ARPU_SEGMENT':[ARPU_SEGMENT],
                            'FREQUENCE':[FREQUENCE],'DATA_VOLUME':[DATA_VOLUME], 
                              'ON_NET':[ON_NET],'MRG':[new_mrs],
                                'REGULARITY':[REGULARITY]})
st.dataframe(input_var)



predicted = model.predict(input_var)
output = None
if predicted[0] == 0:
    output = 'Not Churn'
else:
    output = 'Churn'

prediction, interprete = st.tabs(["Model Prediction", "Model Interpretation"])
with prediction:
    pred = st.button('Push To Predict')
    if pred: 
        st.success(f'The customer is predicted to {output}')
        st.balloons()
# predicter = st.button('Predict')
# if predicter:
#     prediction = model.predict(input_var)
#     output = None
# if prediction[0] == 0:
#     output = 'Not Churn'
# else:
#     output = 'Churn'

#     st.success(f'The Predicted value for your company is {prediction}')
#     st.balloons()
