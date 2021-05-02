import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
matplotlib.use('Agg')
		


def dsm_code_1(avc,df_pred,df_act):

    df = pd.concat([df_pred,df_act],axis=1)
    df['Deviation %'] = (abs(df_act['Actual Gen (MW)'] - df_pred['Predicted Gen (MW)']))/avc*100
    dsm_list = []
    avc_units = avc * 250
    #units_dev = (df['Actual Gen(MW)']-df['Predicted Gen(MW)'])*250
    
    for i in range(len(df)):
        units_dev = abs((df['Actual Gen (MW)'][i]-df['Predicted Gen (MW)'][i]))*250
        dev = df['Deviation %'][i]
        if dev <=15:
            dsm_list.append(0)

        elif dev>15 and dev<=25:
            dsm1 = (units_dev - (0.15*avc_units)) * 0.5
            dsm_list.append(abs(dsm1))

        elif dev>25 and dev<=35:
            dsm1 =  0.1* avc_units * 0.5
            dsm2 = (units_dev - (0.25*avc_units)) * 1
            dsm_list.append(abs(dsm1)+abs(dsm2))

        else:  
            dsm1 =  0.1 * avc_units * 0.5
            dsm2 =  0.1 * avc_units * 1
            dsm3 = (units_dev -(0.35*avc_units)) * 1.5
            dsm_list.append(abs(dsm1)+abs(dsm2)+abs(dsm3))
            

    abc = pd.concat([df,pd.Series(dsm_list)],axis=1)
    abc.columns=['DATE_TIME','BLOCK','Predicted Gen (MW)','Actual Gen (MW)','Deviation %','DSM (Rs.)'] 
    abc = abc[['DATE_TIME','BLOCK','Actual Gen (MW)','Predicted Gen (MW)','Deviation %','DSM (Rs.)']]
    abc[['Actual Gen (MW)','Predicted Gen (MW)','Deviation %','DSM (Rs.)']] = round(abc[['Actual Gen (MW)','Predicted Gen (MW)'
    ,'Deviation %','DSM (Rs.)']],2)
    st.write('')
    st.write('MAPE : {}%'.format(round(abc['Deviation %'].mean(),2)))
    st.write('Total DSM: Rs. {}'.format(round(abc['DSM (Rs.)'].sum(),2)))   

    return abc



def dsm_code_2(avc,df_pred,df_act):

   df = pd.concat([df_pred,df_act],axis=1)
   df['Deviation %'] = (abs(df_act['Actual Gen (MW)'] - df_pred['Predicted Gen (MW)']))/avc*100
   dsm_list = []
   avc_units = avc * 250
    #units_dev = (df['Actual Gen(MW)']-df['Predicted Gen(MW)'])*250
   for i in range(len(df)):
        units_dev = abs((df['Actual Gen (MW)'][i]-df['Predicted Gen (MW)'][i]))*250
        dev = df['Deviation %'][i]
        if dev <= 7:
            dsm_list.append(0)
        elif dev>7 and dev<=15:
            dsm1 = (units_dev - (0.07*avc_units)) * 0.25
            dsm_list.append(abs(dsm1))

        elif dev>15 and dev<=23:
            dsm1 =  0.08 * avc_units * 0.25
            dsm2 = (units_dev - (0.23*avc_units)) * 0.5
            dsm_list.append(abs(dsm1)+abs(dsm2))

        else:  
            dsm1 =  0.08 * avc_units * 0.25
            dsm2 =  0.08 * avc_units * 0.5
            dsm3 = (units_dev -(0.23*avc_units)) * 0.75
            dsm_list.append(abs(dsm1)+abs(dsm2)+abs(dsm3))
   abc = pd.concat([df,pd.Series(dsm_list)],axis=1)
   abc.columns=['DATE_TIME','BLOCK','Predicted Gen (MW)','Actual Gen (MW)','Deviation %','DSM (Rs.)'] 
   abc = abc[['DATE_TIME','BLOCK','Actual Gen (MW)','Predicted Gen (MW)','Deviation %','DSM (Rs.)']]
   abc[['Actual Gen (MW)','Predicted Gen (MW)','Deviation %','DSM (Rs.)']] = round(abc[['Actual Gen (MW)','Predicted Gen (MW)'
    ,'Deviation %','DSM (Rs.)']],2)
   #st.dataframe(abc)
   st.write('')
   st.write('MAPE : {}%'.format(round(abc['Deviation %'].mean(),2)))
   st.write('Total DSM: Rs.{}'.format(round(abc['DSM (Rs.)'].sum(),2)))       

   return abc   
  

    





def visualize():
    uploaded_file_report = st.file_uploader("Upload Forecast Report file", type="csv")
    if uploaded_file_report is not None:
        file_details = {"FileName":uploaded_file_report.name,"FileType":uploaded_file_report.type}
        df_all = pd.read_csv(uploaded_file_report)  
    features = ("Actual Gen (MW)","Predicted Gen (MW)","Deviation %")
    selected_feat = st.multiselect("Features",features,default=["Predicted Gen (MW)","Actual Gen (MW)","Deviation %"])
    
    #st.write(selected_feat[0])
    if st.button('Plot'):
        # fig = plt.figure()
        # sns.lineplot(data=df_all, x="BLOCK", y="Predicted Gen (MW)")
        # sns.lineplot(data=df_all, x="BLOCK", y="Actual Gen (MW)")
        st.dataframe(df_all)
        st.line_chart(df_all[["Predicted Gen (MW)","Actual Gen (MW)","Deviation %"]],use_container_width=True)
        #   st.pyplot(fig)      
	##Upload actual generation file from the plant
	#uploaded_file = st.file_uploader("Upload Actual Generation", type="csv")	
	#if uploaded_file is not None:
   #   file_details = {"FileName":uploaded_file.name,"FileType":uploaded_file.type}
		#  df_all = pd.read_csv(uploaded_file)
    
		

      


		##Load Prediction		
			#df_prediction = pd.read_csv('/Saving_files_here/Predicted_files/1.csv')		
			#fig = plt.figure()
            #sns.countplot(df['species'])
            #st.pyplot(fig)

			## Visualize		
			
      #may_flights = flights.query("month == 'May'")
      
      #plt.plot(df_all['BLOCK'],df_all['Actual Gen (MW)'],label='Actual Gen (MW)')
			#plt.plot(df_all['BLOCK'],df_all['Predicted Gen (MW)'],label='Predicted Gen (MW)')
			#plt.xlabel('Block')
			#plt.ylabel('MW')

			#plt.title("Actual Gen VS Predicted Gen")
			#plt.legend()
			#plt.show()		
			#plt.plot(df_viz[[selected_feat]])	
			#st.write(selected_feat)
					
				#visualize(prediction_df)



	





