# Core Pkgs
import streamlit as st 

# EDA Pkgs
import pandas as pd 
import numpy as np 

# Utils
import os
import joblib 
import hashlib
# passlib,bcrypt

# Data Viz Pkgs
import matplotlib.pyplot as plt 
import matplotlib
matplotlib.use('Agg')

import pickle
import warnings
warnings.filterwarnings("ignore")

#import python scripts

from sklearn.metrics import silhouette_score , silhouette_samples
import seaborn as sns
from sklearn import preprocessing
from sklearn.model_selection import train_test_split


# DB
#from managed_db import *

html_temp = """
		<div style="background-color:{};padding:10px;border-radius:10px">
		<h1 style="color:white;text-align:center;font-family:"Source Sans Pro",sans-serif;">Churn Prediction</h1>
		</div>
		"""

text = """
        <div style="background-color:{};padding:2px;border-radius:5px">
        <h1 style="color:white;text-align:justify; font-size:15px;font-family:"Source Sans Pro",sans-serif;">The telecommunications industry across the world is becoming one of the major sectors and consequently the technical growth and the ever-developing operator number increased the level of competition. Companies are working hard to survive in this competitive market depending on multiple strategies and one of them is to increase the retention period of customers. To apply this strategy, companies must decrease the potential of customer’s churn.Churn refers to the customer movement from one service provider to another</h1>
        </div>
        """

feature_names_best = ['CustomerID','gender', 'SeniorCitizen', 'partner', 'tenure', 'phoneService', 'multipleLine', 'InternetService','OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies','Contract','PaperlessBilling','PaymentMethod','MonthlyCharges','TotalCharges']

gender_dict = {"male":1,"female":2}
feature_dict = {"Yes":1,"No":2}
feature1_dict = {"Yes":1,"No":2,"Other/No Internt Service":3}
feature2_dict = {"ADSL":1,"Fiber":2,"Other":3}
feature3_dict = {"Month-to-Month":1,"2 years":2,"Other":3}
feature4_dict = {"Electronic Cheque":1,"Mail Cheque":2, "Bank Transfer":3, "Other":4}


def get_value(val,my_dict):
	for key,value in my_dict.items():
		if val == key:
			return value 

def get_key(val,my_dict):
	for key,value in my_dict.items():
		if val == key:
			return key

def get_fvalue(val):
	feature_dict = {"Yes":1,"No":2}
	for key,value in feature_dict.items():
		if val == key:
			return value
def get_fvalue(val):
	feature1_dict = {"Yes":1,"No":2,"Other/No Internt Service":3}
	for key,value in feature1_dict.items():
		if val == key:
			return value
def get_fvalue(val):
	feature2_dict = {"ADSL":1,"Fiber":2,"Other":3}
	for key,value in feature2_dict.items():
		if val == key:
			return value
def get_fvalue(val):
	feature3_dict = {"Month-to-Month":1,"2 years":2,"Other":3}
	for key,value in feature3_dict.items():
		if val == key:
			return value
def get_fvalue(val):
	feature4_dict = {"Electronic Cheque":1,"Mail Cheque":2, "Bank Transfer":3, "Other":4}
	for key,value in feature4_dict.items():
		if val == key:
			return value 
        
        # Load ML Models
def load_model(model_file):
	loaded_model = joblib.load(open(os.path.join(model_file),"rb"))
	return loaded_model

# ML Interpretation
#import lime
#import lime.lime_tabular



def main():



	"""Churn Prediction App"""
	#st.title(''<h1 style="float:left;color:white;text-align:left;">"Churn Prediction"</h1>', unsafe_allow_html=True)
	st.markdown(html_temp.format('#120f59'),unsafe_allow_html=True)

	st.markdown("""
	<style>
	.main{color:white;}
	</style>
	""",unsafe_allow_html=True
	)

	menu = ["Home","Classification"]
	#sub_menu = ["Plot","Prediction","Metrics"]

	choice = st.sidebar.selectbox("Menu",menu)
	if choice == "Home":

		st.markdown(
                    """
                        <style> 
                            .main { 
                                background-image: url("https://t3.ftcdn.net/jpg/03/83/81/80/360_F_383818080_RyXyzgCAq9C2Kn6IZkBbt4iU1KGHDYhU.jpg");
                                background-size: cover;
                                } 
                        </style>
                    """,
                     unsafe_allow_html=True
		)


		st.subheader("What is Churn?")
		st.markdown(text.format('background:url("https://png.pngtree.com/thumb_back/fh260/background/20201102/pngtree-global-world-network-and-telecommunication-on-earth-cryptocurrency-and-blockchain-and-image_450389.jpg")'),unsafe_allow_html=True)
                #st.text("Churn refers to the customer movement from one service provider to another")
		
		#st.markdown(descriptive_message_temp,unsafe_allow_html=True)
		#st.image(load_image('images/hepimage.jpeg'))

		st.subheader("Data VS Plot")
		df = pd.read_csv("churn.csv")
		st.dataframe(df)

		st.subheader("Bar Chart - Internet Service ")
		df['InternetService'].value_counts().plot(kind='bar')
		st.pyplot()
		st.set_option('deprecation.showPyplotGlobalUse', False)

		st.subheader("Frequent Distribution Plot - Streaming Tv")
		# Freq Dist Plot
		freq_df = pd.read_csv("churn.csv")
		st.bar_chart(freq_df['StreamingTV'])

		st.subheader("Area Chart - Choose any Feature")
		if st.checkbox("Area Chart"):
                        all_columns = df.columns.to_list()
                        feat_choices = st.multiselect("Choose a Feature",all_columns)
                        new_df = df[feat_choices]
                        st.area_chart(new_df)
		
        
	else:

		st.markdown(
                    """
                        <style> 
                            .main { 
                                background-image: url(""); 
                                background-size: cover; color:white; } 
                        </style>
                    """,
                     unsafe_allow_html=True
		)



		#st.markdown('<h1 style="float:left;color:white;text-align:left;">Curry Kingdom Restaurant</h1>', unsafe_allow_html=True) 
        

		st.subheader("Classification")

		#CustomerID = st.text_input("CustomerID","")
		gender = st.radio("Gender",tuple(gender_dict.keys()))
		SeniorCitizen = st.radio("Senior Citizen",tuple(feature_dict.keys()))
		partner = st.radio("Does the Customer Have a Partner in the Network Registered? ",tuple(feature_dict.keys()))
		tenure = st.slider('Number of months the customer has stayed with the company', min_value=0, max_value=72, value=0)
		phoneService = st.radio("Phone Service",tuple(feature_dict.keys()))
		multipleLine = st.selectbox("Multiple Line",tuple(feature1_dict.keys()))
		InternetService = st.selectbox("Internet Service",tuple(feature2_dict.keys()))
		OnlineSecurity = st.radio("Online Security",tuple(feature1_dict.keys()))
		OnlineBackup = st.radio("Online Backup",tuple(feature1_dict.keys()))
		DeviceProtection = st.radio("DeviceProtection",tuple(feature1_dict.keys()))
		TechSupport = st.radio("Tech Support",tuple(feature1_dict.keys()))
		StreamingTV = st.radio("StreamingTV",tuple(feature1_dict.keys()))
		StreamingMovies = st.radio("StreamingMovies",tuple(feature1_dict.keys()))
		Contract = st.selectbox("Contract",tuple(feature3_dict.keys()))
		PaperlessBilling= st.selectbox("Paperless Billing",tuple(feature_dict.keys()))
		PaymentMethod= st.selectbox("Payment Method",tuple(feature4_dict.keys()))
		MonthlyCharges= st.number_input("Monthly Charges",min_value=0, max_value=10000, value=0)
		TotalCharges= st.number_input("Total Charges",min_value=0.00, max_value=10000.00, value=0.00)
                    
		feature_list = [get_value(gender,gender_dict),get_fvalue(SeniorCitizen),get_fvalue(partner),tenure,get_fvalue(phoneService),get_fvalue(multipleLine),get_fvalue(InternetService),get_fvalue(OnlineSecurity),get_fvalue(OnlineBackup),get_fvalue(DeviceProtection),get_fvalue(TechSupport),get_fvalue(StreamingTV),get_fvalue(StreamingMovies),get_fvalue(Contract),get_fvalue(PaperlessBilling),get_fvalue(PaymentMethod),get_fvalue(MonthlyCharges),get_fvalue(TotalCharges)]
		#st.write(len(feature_list))
		#st.write(feature_list)
		data = {"gender":gender, "SeniorCitizen":SeniorCitizen, "partner":partner, "tenure":tenure, "phoneService":phoneService, "multipleLine":multipleLine, "InternetService":InternetService,"OnlineSecurity":OnlineSecurity, "OnlineBackup":OnlineBackup, "DeviceProtection":DeviceProtection, "TechSupport":TechSupport, "StreamingTV":StreamingTV, "StreamingMovies":StreamingMovies,"Contract":Contract,"PaperlessBilling":PaperlessBilling,"PaymentMethod":PaymentMethod,"MonthlyCharges":MonthlyCharges,"TotalCharges":TotalCharges}
                                    

                                
		features_df = pd.DataFrame.from_dict([data])

		#drop_vars = ["CustomerID"]
		#features_df = features_df.drop(drop_vars, axis=1)
                
		st.markdown("<h3></h3>", unsafe_allow_html=True)

		st.write('Overview of input is shown below')
		st.markdown("<h3></h3>", unsafe_allow_html=True)
		st.dataframe(features_df)

		result = ""

		#st.json(pretty_result)
		#single_sample = np.array(feature_list).reshape(1,-1)

                #Preprocessing
		#read the data set
		#df = pd.read_csv("churn.csv")
		#df.head()

		#for col in df.columns:
		    #print('{} : {}'.format(col,df[col].unique()))


                #find the unique value
		#features_df['customerID'].unique()


                #drop duplicates
		#features_df.drop_duplicates()
		#features_df.drop_duplicates(subset=['customerID'])

                #drop columns
		#df.drop(['Dependents'], axis=1, inplace=True)


                #convert categorical values into numeric values

		def getGender(str):
                    if str=="Male":
                        return 1
                    else:
                        return 2
                    
		def getPartner(str):
                    if str=="Yes":
                        return 1
                    else:
                        return 2
                                    
		def getPhoneService(str):
                    if str=="Yes":
                        return 1
                    else:
                        return 2
		def getMultipleLines(str):
                    if str=="Yes":
                        return 1
                    elif str == "No":
                        return 2
                    else:
                        return 3
		def getInternetService(str):
                    if str=="DSL":
                        return 1
                    elif str == "Fiber optic":
                        return 2
                    else:
                        return 3
		def getOnlineSecurity(str):
                    if str=="Yes":
                        return 1
                    elif str == "No":
                        return 2
                    else:
                        return 3
		def getOnlineBackup(str):
                    if str=="Yes":
                        return 1
                    elif str == "No":
                        return 2
                    else:
                        return 3
		def getDeviceProtection(str):
                    if str=="Yes":
                        return 1
                    elif str == "No":
                        return 2
                    else:
                        return 3
		def getTechSupport(str):
                    if str=="Yes":
                        return 1
                    elif str == "No":
                        return 2
                    else:
                        return 3
		def getStreamingTV(str):
                    if str=="Yes":
                        return 1
                    elif str == "No":
                        return 2
                    else:
                        return 3
		def getStreamingMovies(str):
                    if str=="Yes":
                        return 1
                    elif str == "No":
                        return 2
                    else:
                        return 3
		def getContract(str):
                    if str=="Month-to-month":
                        return 1
                    elif str == "Two year":
                        return 2
                    else:
                        return 3
		def getPaperlessBilling(str):
                    if str=="Yes":
                        return 1
                    else:
                        return 2
		def getPaymentMethod(str):
                    if str=="Electronic check":
                        return 1
                    elif str == "Mailed check":
                        return 2
                    elif str == "Bank transfer (automatic)":
                        return 3
                    else:
                         return 4
                    
		features_df["gender"]= features_df["gender"].apply(getGender)
		features_df["partner"]= features_df["partner"].apply(getPartner)
                # df["Dependents"]= df["Dependents"].apply(getDependents)
		features_df["phoneService"]= features_df["phoneService"].apply(getPhoneService)
		features_df["multipleLine"]= features_df["multipleLine"].apply(getMultipleLines)
		features_df["InternetService"]= features_df["InternetService"].apply(getInternetService)
		features_df["OnlineSecurity"] = features_df["OnlineSecurity"].apply(getOnlineSecurity)
		features_df["OnlineBackup"]= features_df["OnlineBackup"].apply(getOnlineBackup)
		features_df["DeviceProtection"]= features_df["DeviceProtection"].apply(getDeviceProtection)
		features_df["TechSupport"]= features_df["TechSupport"].apply(getOnlineBackup)
		features_df["StreamingTV"]= features_df["StreamingTV"].apply(getStreamingTV)
		features_df["StreamingMovies"]= features_df["StreamingMovies"].apply(getStreamingMovies)
		features_df["Contract"]= features_df["Contract"].apply(getContract)
		features_df["PaperlessBilling"]= features_df["PaperlessBilling"].apply(getPaperlessBilling)
		features_df["PaymentMethod"] = features_df["PaymentMethod"].apply(getPaymentMethod)

                #replace SeniorCitizen value 0 by 2
		#features_df['SeniorCitizen']=features_df['SeniorCitizen'].replace(to_replace=0,value=2)

                #Convert TotalCharges into float
		features_df['TotalCharges'] = features_df['TotalCharges'].apply(pd.to_numeric, errors='coerce')
		features_df['MonthlyCharges'] = features_df['MonthlyCharges'].apply(pd.to_numeric, errors='coerce')

                #fininding the sum of null values
		#sum(df.isnull().sum())

                #fininding the sum of null values in TotalCharges
		#features_df['TotalCharges'].isnull().sum()

                #Replace null values with mean
		#features_df.fillna(df.mean(), inplace=True)

                #Standardize Numeric Variables

		def starndadize_data(df):
                    '''Starndardize the numerical values in the input data frame
                    Args:
                       df: data frame that need to standardize
                    Returns:
                       df: data frame with starndardize numerical variables
                    '''
                   
                    print('Assign the numeric values list in the data frame into num_col')     
                    num_col = list(df.select_dtypes(include=np.number).columns)
                    
                    print('Replace null values with Zero')
                    df[num_col] = df[num_col].fillna(0)
                   
                    print('Starndadize the numerical variables')
                    df[num_col] = preprocessing.StandardScaler().fit_transform(df[num_col])
                    
                    return df



		preprocess_output = starndadize_data(features_df)

		# load model
	
		model = joblib.load(r"./best_model.sav")
		if st.button('predict'):

                        

                
                        #y=preprocess_output['Churn']
                        #x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.3)


                        
                        #get the prediction

                        pred = model.predict(preprocess_output)
                        x_test = pd.DataFrame(pred,columns=["Predictions"])
                        x_test = x_test.replace({1: 'Yes, the Customer will chain',2:'No, Customer is satisfy with out service'})

                        st.markdown("<h3></h3>",unsafe_allow_html=True)
                        st.subheader('Prediction')
                        st.write(x_test)

                    
                   
if __name__ == '__main__':
	main()



	
    
