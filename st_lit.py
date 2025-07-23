import streamlit as st
import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt 
import seaborn as sns
from tensorflow.keras.models import load_model
import joblib

#python -m streamlit run st_lit.py

st.set_page_config(page_title='Credit Risk App', layout='wide')

def login():
    st.title('Login Page')
    password=st.text_input('Enter Password',type='password')
    if st.button('Logon'):
        if password=='29720051414@#':
            st.session_state.logged_in=True
            st.success('login Successful')
        else:
            st.error('wrong_password')


def display_model():
    model=load_model('my_bank_model.h5')
    st.title('Credit Risk Prediction 📊 ')
    Duration_of_Credit_Credit_monthly=st.number_input('Duration_Of_Credit_Credit_Monthly',step=1.0)
    Credit_Amount=st.number_input('Credit_Amount',step=1.0)
    Age_years=st.number_input('Age_years',step=1.0)
    Duration_in_Current_address=st.number_input('Duration_in_Current_address',step=1.0)

    Account_Balance=st.selectbox('Account_Balance',
                             ['No_Current_Account_Found',
                             'Less_than_200_DM',
                             'From_200_To_999_DM',
                           'More_than_1000_DM'])

    Purpose=st.selectbox('Purpose',
    ['unknown',
     'New_car',
     'Used_car',
     'Furniture',
     'Home_appliances',
     'Tv_or_Radio',
     'Tuition_fees',
     'Vacation',
     'personal_reason','Ather'])

    Value_Savings_Stocks=st.selectbox('Value_Savings_Stocks',
      ['less_than_100_DM',
     'From_100_To_499_DM',
     'Form_500_To_999_DM',
     'More_than_1000_DM',
     'Unknown'])

    Length_of_current_employment=st.selectbox('Length_of_current_employment',
     ['Less_than_year',
      'From_1year_to_4years',
      'From_4years_to_7years',
      'More_than_7years',
      'Unemployed'])

    Instalment_per_cent=st.selectbox('Instalment_per_cent',
      ['20% of income',
       '25% of income',
      '30% of income',
      '35% of income'])
 

    Sex_Marital_Status=st.selectbox('Sex_Marital_Status',
     ['Single_male',
      'Married_female',
      'Married_male','Divorced'])


    Guarantors=st.selectbox('Guarantors',
     ['None',
       'Partner',
      'Other Guarantor'])


    Most_valuable_available_asset=st.selectbox('Most_valuable_available_asset',
     ['car',
      'Property_Ownership',
       'Life_Insurance',
      'Nothing'])


    Concurrent_Credits=st.selectbox('Concurrent_Credits',
                                ['Nothing',
                                 'Store',
                                 'Another_bank'])


    Type_of_apartment=st.selectbox('Type_of_apartment',['Rent'
                                                    ,'Joint_ownership',
                                                    'Private_ownership'])


    Occupation=st.selectbox('Occupation',['Unemployed'
                                      ,'simple_employee',
                                      'Good employee',
                                      'Highly paid worker'])

    Payment_Status_of_Previous_Credit=st.selectbox('Payment_Status_of_Previous_Credit',['No Credit',
                                                                                    'Paid on Time',
                                                                                    'Paid in Full',
                                                                                    'Slight Delay',
                                                                                    'Critical / Default'])

    Telephone=st.selectbox('Telephone',['Yes','NO'])
    Foreign_Worker=st.selectbox('Foreign_Worker',['Yes','No'])


    input_dict={
     'Account_Balance':Account_Balance,
     'Duration_of_Credit_Credit_monthly':Duration_of_Credit_Credit_monthly,
     'Payment_Status_of_Previous_Credit':Payment_Status_of_Previous_Credit,
     'Purpose':Purpose,
     'Credit_Amount' :Credit_Amount,
     'Value_Savings_Stocks':Value_Savings_Stocks,
     'Length_of_current_employment':Length_of_current_employment,
     'Instalment_per_cent': Instalment_per_cent,
     'Sex_Marital_Status':Sex_Marital_Status,
     'Guarantors':Guarantors,
     'Duration_in_Current_address':Duration_in_Current_address,
     'Most_valuable_available_asset':Most_valuable_available_asset,
     'Age_years':Age_years,
     'Concurrent_Credits':Concurrent_Credits,
     'Type_of_apartment':Type_of_apartment,
     'Occupation':Occupation,
     'Telephone':Telephone,
     'Foreign_Worker':Foreign_Worker
    } 

    df_input=pd.DataFrame([input_dict])

    df_cat = df_input.select_dtypes(include='object')
    df_num = df_input.select_dtypes(exclude='object')

    encoder=joblib.load('encoder.joblib')
    encoded_array=encoder.transform(df_cat)
    encoded_df = pd.DataFrame(encoded_array, columns=encoder.get_feature_names_out(df_cat.columns))
    encoded_df.index=df_input.index
    df_final = pd.concat([encoded_df, df_num], axis=1)

    prediction=model.predict(df_final)[0][0]
 
    st.subheader('Result')
    btn=st.button('Predict')
    if btn:
      if prediction >=0.5:
        st.success(' Qualified ✅ ')
      else:
        st.error('Unqualified ❌ ')
      
def display_analysis():
    df=pd.read_csv('german.csv')
    df['Purpose'].replace(
    {0:'unknown',
     1:'New_car',
     2:'Used_car',
     3:'Furniture',
     4:'Home_appliances',
     5:'Tv_or_Radio',
     6:'Tuition_fees',
     8:'Vacation',
     9:'personal_reason',
     10:'Ather'},
    inplace=True)
    
    df['Account_Balance'].replace(
    {1:'No_Current_Account_Found',
     2:'Less_than_200_DM',
     3:'From_200_To_999_DM',
     4:'More_than_1000_DM'},
    inplace=True)
    
    df['Sex_Marital_Status'].replace(
    {1:'Single_male',
     2:'Married_female',
     3:'Married_male',
     4:'Divorced'},
    inplace=True)

    df['Occupation'].replace(
    {1:'Unemployed',
     2:'simple_employee',
     3:'Good employee',
     4:'Highly paid worker'},
    inplace=True)
    
    df['Instalment_per_cent'].replace(
    {1: '20% of income',
     2: '25% of income',
     3: '30% of income',
     4: '35% of income'},
    inplace=True)
     
    df['Creditability'].replace({0:'Unqualified',1:'Qualified'},inplace=True)
    palette = {'Unqualified': '#0D1B2A', 'Qualified': "#C9DEE48E"}

    st.markdown('Loan Purpose vs Creditability')
    st.markdown('This graph explains the relationship  between Loan Purpose and whether a person is Qualified or Unqualified')

    
    fig,ax=plt.subplots()
    fig.tight_layout()
    sns.countplot(data=df,x='Purpose',ax=ax,palette=palette,hue='Creditability')
    ax.set_title('Frequency of each loan purpose with creditability',fontsize=14)
    ax.set_xlabel('Loan Purpose', fontsize=12)
    ax.set_ylabel('Count',fontsize=12)
    plt.xticks(rotation=45)
    plt.grid(True)
    st.pyplot(fig)
    
    st.markdown('#--------------')
    st.markdown('Instatment_per_cent vs Creditability')
    st.header("The percentage of the monthly instalment compared to the loan amount or the person's income.")
    fig,ax=plt.subplots()
    sns.countplot(data=df,x='Instalment_per_cent',palette=palette,hue='Creditability',ax=ax)
    ax.set_title('Frequency of each Instalment_per_cent with Creditability')
    ax.set_xlabel('Instalment_per_cent',fontsize=12)
    ax.set_ylabel('Count')
    plt.xticks(rotation=45)
    plt.grid(True)
    st.pyplot(fig)     
    
    st.markdown('#--------------')
    st.markdown('Laon Purpose vs Account_balance')
    st.markdown('This graph explains the relationship between Loan Purpose and whether a person is qualified or unqualified ')
    
    fig ,ax=plt.subplots()
    fig.tight_layout()
    sns.countplot(data=df,x='Account_Balance',hue='Creditability',ax=ax,palette=palette)
    ax.set_title('frequency of each account balance with Creditability ',fontsize=14)
    ax.set_xlabel('Account_balance',fontsize=12)
    ax.set_ylabel('Count',fontsize=12)
    plt.xticks(rotation=45)
    plt.grid(True)
    st.pyplot(fig)
    
    st.markdown("-----------------")
    st.markdown('Marital column frequency distibution')
    st.markdown('This graph shows frequency distribution for Sex & Marital Status')
    
    fig,ax=plt.subplots()
    fig.tight_layout()
    sns.countplot(data=df,x='Sex_Marital_Status',ax=ax, color='black')
    ax.set_title('frequency to each value in Sex_Marital_Status cloumn')
    ax.set_xlabel('Sex_Marital_Status')
    ax.set_ylabel('Count')
    plt.xticks(rotation=45)
    plt.grid(True)
    st.pyplot(fig)
    
    st.markdown('-----------')
    
    df['Creditability'].replace({'Unqualified':0,'Qualified':1},inplace=True)
    st.subheader('Data_Preview')
    st.write(df.head())
    st.markdown("-----------------")

    st.markdown("*Age VS Loan Approvel Visualization ---->\n This graph shows how the continuous variable (e.g., Age) varies across the binary target classes (e.g., Creditability = 0 or 1). It helps us understand whether there's a noticeable difference in average values between the two classes.")


    fig,ax=plt.subplots()
    fig.tight_layout()
    sns.stripplot(x='Creditability',y='Age_years',data=df,ax=ax,jitter=True)
    ax.set_title('Relationship between Age and Loan Approval status')
    ax.set_xlabel('loan_stutas')
    ax.set_ylabel('Age')
    st.pyplot(fig)
    
def welcome():
    st.markdown('# welcome')

    st.markdown('''This is my first projecct in streamlit and i hope you like it ''')
    
    st.markdown('''This project is about credit risk predection, which is a crucial aspect of the banking and financial industry.
                The goal is to predict whether a Loan Application will be approved or not based on various features such as credit history,
                income, and other financial indicators. The project uses a deep learning model to make these predictions.''')
    st.markdown('''The project includes a user-friendly interface built with Streamlit,
                allowing users to input loan appliction details and receive predictions on whether the application is likely to be approved or not.''')
    
    st.markdown('''The project also includes data analysis and visualization components to help users understand the factors influencing loan approval decisions''')
        
    st.markdown('''The project is built using python and leverages popular libraries such as Tensorflow, pandas, numpy, matplotlib, seaborn, and scikit-learn for data processing, model training, and visualization.''')
    
    st.markdown('''The project is agreat example of how machine learning can be applied to real-world problems in the financial scector,
                helping banks and financial institutions make more infiormed decisions about loan applictions.''')
    
    
# This is the main function that controls the flow of the application 
def main_appli():
    st.sidebar.title('📁 Navigation')
    page=st.sidebar.radio('Go to',['Welcome','display_model','display_analysis'])
    if page=='Welcome':
        welcome()
    elif page=='display_model':
        display_model()
    elif page=='display_analysis':
        display_analysis()
        

def main():
    if 'logged_in' not in st.session_state:
        st.session_state.logged_in=False
    if st.session_state.logged_in:
        main_appli()
    else:
        login()
        
main()