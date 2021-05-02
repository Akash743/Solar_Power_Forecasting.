import pandas as pandas
import numpy as np
import joblib
import os
import streamlit as st
#from sklearn.preprocessing import StandardScaler

outlier_imputer_dict = np.load('outlier_imputer_dict.npy',allow_pickle=True)
def outlier_imputer(df):
    		for var in df.columns:
        		df[var] = np.where(df[var] > outlier_imputer_dict[var]['99th'],outlier_imputer_dict[var]['99th'],df[var])
        		df[var] = np.where(df[var] < outlier_imputer_dict[var]['1st'],outlier_imputer_dict[var]['1st'],df[var])
        
    		return df    

@st.cache(allow_output_mutation=True)
def load_model(model_file):
	loaded_model = joblib.load(model_file)
	return loaded_model	

def run_ml_app(df):

	#a df = outlier_imputer(df)
	model = load_model("Solar_forecast_model_final.pkl")
	df_pred = model.predict(df)

	return df_pred

	