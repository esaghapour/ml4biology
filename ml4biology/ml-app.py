import pandas as pd
import streamlit as st
import random
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from missingpy import MissForest 
from sklearn.impute import KNNImputer
import base64
from sklearn.decomposition import PCA
import random
from sklearn.manifold import TSNE
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import RepeatedKFold
from sklearn import preprocessing
from sklearn.linear_model import LassoCV
from fancyimpute import SoftImpute
from fancyimpute import IterativeImputer as MICE
from Missxgb import Missxgb
import misslgb
from gain import gain 
import Missxgb

import lightgbm as lgb
st.title('Machine Learning for Biologists')


# @st.cache
def file_selector():
    file = st.sidebar.file_uploader("Choose a CSV file", type="csv")
    if file is not None:
      data = pd.read_csv(file)
      return data
    else:
      st.sidebar.text("Please upload a csv file")
uploaded_file=file_selector()

# print(uploaded_file)
  
if uploaded_file is not None:   
    target_options = uploaded_file.columns
    chosen_target = 'Target'
    labels = uploaded_file['Target'].values
    labels=(labels-np.min(labels))/(np.max(labels)-np.min(labels))
    # Standardize the feature data
    data = uploaded_file.loc[:, uploaded_file.columns != chosen_target]
    count_nan_in_data = data.isnull().sum()/len(data)
    colnamess = data.columns
    
    data=data.values
    
    
    plt.style.use('ggplot')
        
    
    col1, col2 = st.beta_columns(2)
    
    # st.header("Histogram ")
    col1.header("Histogram")
    
    fig, ax = plt.subplots()
    # Add titles
    plt.bar(colnamess,np.array(count_nan_in_data)*100, color='red')
    # plt.title("Histogram")
    plt.xlabel("Columns")
    plt.ylabel("Percent of missing rate")
    ax.set_xticks(colnamess)    # This ensures we have one tick per year, otherwise we get fewer
    ax.set_xticklabels(colnamess, rotation='vertical')
    # px.bar(colnamess,count_nan_in_data)
    # datas = diabetes.data
    
    col1.pyplot(fig)
    # st.stop()
    plt.close()
    
    # st.header("Correlation between 2 features")
    st.sidebar.header("Correlation")
    columnx = st.sidebar.selectbox("Column X", colnamess)
    # pic = st.sidebar.selectbox("Correlation X", list(pics3.keys()), 0) 
    col2.header("Correlation")
    imp_median = SimpleImputer(missing_values=np.nan, strategy='median')
    X_imputed=imp_median.fit_transform(data)
    columny = st.sidebar.selectbox("Column Y", colnamess,1)
    print(columny)
    print(columnx)
    fig, ax = plt.subplots()
    # plt.show()
    # fig.show()
    ax.scatter(X_imputed[:,np.where(colnamess==columnx)],X_imputed[:,np.where(colnamess==columny)])
    # plt.title("Correlation")
    plt.xlabel(columnx)
    plt.ylabel(columny)
    col2.pyplot(fig)
    plt.close()

