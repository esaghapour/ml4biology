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

####################################
    def get_imouted_data(X_imputed,colnamess, names):
            
            st.write("It's ready to downlaod! " + names + " model.")
            df = pd.DataFrame(X_imputed, columns=colnamess)
            csv = df.to_csv(index=False)
            b64 = base64.b64encode(csv.encode()).decode()  # some strings <-> bytes conversions necessary here
            href = f'<a href="data:file/csv;base64,{b64}">Download the Result as CSV File</a> (right-click and save as &lt;some_name&gt;.csv)'  
            st.markdown(href, unsafe_allow_html=True)
    
    def get_param_lgbm():
        min_child_weight = st.sidebar.slider('min_child_weight', 0.0, 1.0, .03)
        feature_fraction = st.sidebar.slider('feature_fraction', 0.0, 1.0, .03)
        learning_rate = st.sidebar.slider('learning_rate', 0.0, 1.0, 0.1)
        subsample= st.sidebar.slider('subsample',0.0, 1.0, .6)
        min_data_in_leaf = st.sidebar.slider('min_data_in_leaf:' , 1,10,5)
        bagging_fraction = st.sidebar.slider('bagging_fraction', 0.0, 1.0, 0.4)
        reg_alpha = st.sidebar.slider('reg_alpha :', 0.0, 1.0, .3)
        reg_lambda = st.sidebar.slider('reg_lambda :', 0.0, 1.0, .6)
        n_estimators = st.sidebar.slider('n_estimators', 10, 1000, 100)
        max_iter=st.sidebar.slider('max_iter (works for part 1)',1, 100, 10)

        params = {            'min_child_weight':min_child_weight,
                              'feature_fraction': feature_fraction,
                              'bagging_fraction': bagging_fraction,
                              'min_data_in_leaf': min_data_in_leaf,
                              'objective': 'regression',
                              'max_depth': -1,
                              'learning_rate':learning_rate,
                              "boosting_type": "gbdt",
                              "bagging_seed": 11,
                              "metric": 'mse',
                              "verbosity": -1,
                              'reg_alpha':reg_alpha,
                              'reg_lambda':reg_lambda,
                              'subsample': subsample,
                              'n_estimators' :n_estimators,
                              'max_iter' : max_iter
                              }
        return params
    @st.cache
    def hold_data(X_imputed,colnamess):
        X_imputed=pd.DataFrame(X_imputed)
        # X_imputed=X_imputed.rename(columns=colnamess)
        X_imputed.columns=colnamess
        X_imputed.to_csv('x_imputed.csv', index=False)
        

    
    X_imputed=[]

    imp_mean = SimpleImputer(missing_values=np.nan, strategy='mean')
    imp_median = SimpleImputer(missing_values=np.nan, strategy='median')
    
    
    st.header("Imputation algorithm")
    st.sidebar.header("Imputation algorithm")
    
    imputers =[
        "My data do not need to impute!" ,  
        "Mean",
        "Median",
        "Missforest",
        "KNNImputer",
        "Misslgbm",
        "Misxgboost",
        "GAIN",
        "SoftImpute",
        "MICE",
        "All" ]
    # imputers = st.sidebar.selectbox("Imutation algorithm", imputers_methods,0)
    imputers = st.sidebar.selectbox(
        label="Choose...", options=imputers
    )
    print(imputers)
    # # st.image(pics[pic], use_column_width=True, caption=pics[pic])
    if imputers=="My data do not need to impute!":
        st.write("OK!")
        
    elif imputers=="Missforest":
        min_samples_split = st.sidebar.slider('min_samples_split', 2, 10, 2)
        min_samples_leaf = st.sidebar.slider('min_samples_leaf', 0, 10, 1)
        min_weight_fraction_leaf = st.sidebar.slider('min_weight_fraction_leaf', 0.0, 1.0, 0.0)
        max_iter=st.sidebar.slider('max_iter',1, 100, 10)
        n_estimators = st.sidebar.slider('n_estimators:' ,10,1000,100,10)
        min_impurity_decrease = st.sidebar.slider('min_impurity_decrease', 0.0, 1.0, 0.0)
    
        params = {   'min_samples_split':min_samples_split,
                      'min_samples_leaf': min_samples_leaf,
                      'min_weight_fraction_leaf': min_weight_fraction_leaf,
                      'max_iter': max_iter,
                      'n_estimators': n_estimators,
                      'min_impurity_decrease': min_impurity_decrease,
                    }
        # imp_forest = MissForest(random_state=1337,n_estimators=parameters)
        # X_imputed0 = imp_lgb.fit_transform(data)
        st.write("It's done by Missforest method!")
    
    elif imputers=="Misslgbm":
        params = get_param_lgbm()
        
        # imp_lgb = Misslgb(random_state=1337)
        # X_imputed0 = imp_lgb.fit_transform(data)
        st.write("It's done by Mislighbm method!")
    
    elif imputers=="Misxgboost":
        max_iter=st.sidebar.slider('max_iter',1, 100, 10)
        n_estimators = st.sidebar.slider('n_estimators', 10, 1000, 100)
        learning_rate = st.sidebar.slider('learning_rate', 0.0, 1.0, 0.1)
        min_child_weight = st.sidebar.slider('min_child_weight', 0.0, 5.0, 1.5)
        subsample= st.sidebar.slider('subsample',0.0, 1.0, .6)
        max_depth = st.sidebar.slider('max_depth:', 1,10,3)
        gamma = st.sidebar.slider('gamma', 0.0, 1.0, 0.0)
        reg_alpha = st.sidebar.slider('reg_alpha :', 0.0, 100.0, 0.0)
        reg_lambda = st.sidebar.slider('reg_lambda :', 0.0, 100.0, 0.0)
        colsample_bytree = st.sidebar.slider('colsample_bytree :', 0.0, 1.0, 0.4)
        seed = st.sidebar.slider('seed :', 0, 100, 42)
    
        params = {           'colsample_bytree':colsample_bytree,
                              'gamma':gamma,                 
                              'learning_rate':learning_rate,
                              'max_depth':max_depth,
                              'min_child_weight':min_child_weight,
                              'n_estimators':n_estimators,                                                                    
                              'reg_alpha':reg_alpha,
                              'reg_lambda':reg_lambda,
                              'subsample':subsample,
                              'seed':seed,
                              'max_iter':max_iter
                              }
        
        # imp_xgb = Missxgb(random_state=1337,parameters)
        # X_imputed0 = imp_xgb.fit_transform(data2)
        st.write("It's done by Misxgboost method!")
    
    elif imputers=="GAIN":
    
        batch_size = st.sidebar.slider('Batch_size : (2 ^ Value)', 0, 8, 1)
        hint_rate = st.sidebar.slider('Hint_rate', 0.0, 1.0, 0.9,.1)
        alpha = st.sidebar.slider('Alpha', 0, 200, 100)
        iterations= st.sidebar.slider('Iterations',5000, 20000, 10000,100)
        
        gain_parameters = {'batch_size': 2**batch_size,
                          'hint_rate':hint_rate,
                          'alpha': alpha,
                          'iterations': iterations}
        print(gain_parameters)
        # X_imputed0=gain(np.array(data2), gain_parameters)
        st.write("It's done by GAIN method!")
    
    elif imputers=="SoftImpute":
        sepal_length = st.sidebar.slider('Number of trees', 10, 1000, 100)
        # X_imputed5=IterativeSVD().fit_transform(data2+.00001)
        st.write("It's done by SoftImpute method!")
    
    elif imputers=="Median":
        # imp_mean = SimpleImputer(missing_values=np.nan, strategy='mean')
        X_imputed=imp_median.fit_transform(data)
        st.write("It's done by Median method!")
    
    elif imputers=="Mean":
        # imp_median = SimpleImputer(missing_values=np.nan, strategy='median')
        X_imputed=imp_mean.fit_transform(data)
        st.write("It's done by Mean method!")
    
    elif imputers=="KNNImputer":
        parameters = st.sidebar.slider('Number of neighbors', 1, 10, 5)
        # imp_knn = KNNImputer(n_neighbors=parameters)
        # X_imputed=imp_knn.fit_transform(data)
        st.write("It's done by KNNImputer method!")
    
    
    btn = st.sidebar.button("Run-part-1")
    if btn:
        
        if imputers=="My data do not need to impute!":
            st.write("OK!")
        elif imputers=="Missforest":    
            imp_forest = MissForest(min_samples_split=params.get('min_samples_split'),
                      min_samples_leaf=params.get('min_samples_leaf'),
                      min_weight_fraction_leaf=params.get('min_weight_fraction_leaf'),
                      max_iter=params.get('max_iter'),
                      n_estimators=params.get('n_estimators'),
                      min_impurity_decrease=params.get('min_impurity_decrease'))
            X_imputed = imp_forest.fit_transform(data)
            st.write("It's done by Missforest method!")
            get_imouted_data(X_imputed,colnamess, imputers)
            X_imputed=hold_data(X_imputed,colnamess)
        
        elif imputers=="Misslgbm":
            imp_lgb = misslgb.Misslgb(min_child_weight=params.get('min_child_weight'),
                              feature_fraction=params.get('feature_fraction'),
                              bagging_fraction=params.get('bagging_fraction'),
                              min_data_in_leaf=params.get('min_data_in_leaf'),
                              learning_rate=params.get('learning_rate'),
                              reg_alpha=params.get('reg_alpha'),
                              reg_lambda=params.get('reg_lambda'),
                              # subsample=params.get( 'subsample'),
                              n_estimators =params.get('n_estimators'),
                              max_iter =params.get('max_iter'))
            X_imputed = imp_lgb.fit_transform(data)
            st.write("It's done by Misslgbm method!")
            get_imouted_data(X_imputed,colnamess, imputers)
            X_imputed=hold_data(X_imputed,colnamess)
    
        elif imputers=="Misxgboost":
            
            imp_xgb = Missxgb.Missxgb(random_state=params.get('random_state'),
                              min_child_weight=params.get('min_child_weight'),
                              colsample_bytree=params.get('colsample_bytree'),
                              gamma=params.get('gamma'),                 
                              learning_rate=params.get('learning_rate'),
                              max_depth=params.get('max_depth'),
                              n_estimators=params.get('n_estimators'),                                                                    
                              reg_alpha=params.get('reg_alpha'),
                              reg_lambda=params.get('reg_lambda'),
                              subsample=params.get('subsample'),
                              seed=params.get('seed'),
                              max_iter=params.get('max_iter') )
            X_imputed = imp_xgb.fit_transform(data)
            st.write("It's done by Missforest method!")
            get_imouted_data(X_imputed,colnamess, imputers)
            X_imputed=hold_data(X_imputed,colnamess)
        
        elif imputers=="GAIN":
        
            X_imputed=gain(np.array(data), gain_parameters)
            st.write("It's done by GAIN method!")
            get_imouted_data(X_imputed,colnamess, imputers)
            X_imputed=hold_data(X_imputed,colnamess)

    
        elif imputers=="SoftImpute":
            
            X_imputed=SoftImpute().fit_transform(data)
            st.write("It's done by SoftImpute method!")
            get_imouted_data(X_imputed,colnamess, imputers)
            X_imputed=hold_data(X_imputed,colnamess)

        
        elif imputers=="Median":
            # imp_mean = SimpleImputer(missing_values=np.nan, strategy='median')
            X_imputed=imp_median.fit_transform(data)
            st.write("It's done by Median method!")
            get_imouted_data(X_imputed,colnamess, imputers)
            X_imputed=hold_data(X_imputed,colnamess)
    
        elif imputers=="Mean":
            X_imputed=imp_mean.fit_transform(data)
            st.write("It's done by Mean method!")
            get_imouted_data(X_imputed,colnamess, imputers)
            X_imputed=hold_data(X_imputed,colnamess)

    
        elif imputers=="KNNImputer":
            imp_knn = KNNImputer(n_neighbors=parameters)
            X_imputed=imp_knn.fit_transform(data)
            get_imouted_data(X_imputed,colnamess,imputers)
            X_imputed=hold_data(X_imputed,colnamess)

    
        else:
            imp_xgb = Missxgb(random_state=1337)
            X_imputed1=imp_xgb.fit_transform(data)
            get_imouted_data(X_imputed1,colnamess, "Misxgboost")
    
            imp_lgb = misslgb(random_state=1337)
            X_imputed1=imp_lgb.fit_transform(data)
            get_imouted_data(X_imputed1,colnamess, "Misslgbm")
            X_imputed.append(X_imputed1)
            
            imp_forest = MissForest(random_state=1337)
            X_imputed1=imp_forest.fit_transform(data)
            get_imouted_data(X_imputed1,colnamess, "MissForest")
            X_imputed.append(X_imputed1)
    
            # imp_forest = MissForest(random_state=1337)
            imp_knn = KNNImputer(n_neighbors=5)
            X_imputed1=imp_knn.fit_transform(data)
            get_imouted_data(X_imputed1,colnamess, "KNNImputer")
            X_imputed.append(X_imputed1)
            
            X_imputed1=imp_median.fit_transform(data)
            get_imouted_data(X_imputed1,colnamess, "Median")
            X_imputed.append(X_imputed1)
            
            X_imputed1=imp_mean.fit_transform(data)
            get_imouted_data(X_imputed1,colnamess, "Mean")
            X_imputed.append(X_imputed1)
            # imp_median.fit_transform(data)
            
            X_imputed1=MICE().fit_transform(data)
            get_imouted_data(X_imputed1,colnamess, "MICE")
            X_imputed.append(X_imputed1)
            gain_parameters = {'batch_size': 32,
                          'hint_rate': .9,
                          'alpha': 100,
                          'iterations': 10000}
            X_imputed1=gain(np.array(data), gain_parameters)
            get_imouted_data(X_imputed1,colnamess, "GAIN")
            X_imputed.append(X_imputed1)
            
            X_imputed1=SoftImpute().fit_transform(data)
            get_imouted_data(X_imputed1,colnamess, "SoftImpute")
            X_imputed.append(X_imputed1)
    
            st.write("It's done for all methods!")


# #################### Cluster part ##########################################
# print(X_imputed)
    X_imputeds=np.array(pd.read_csv('x_imputed.csv'))
    print(X_imputeds)
    
    st.header("Cluster algorithm")
    st.sidebar.header("Cluster algorithm")
    
    clustering_methods =[
        "My data do not need to cluster!" ,  
        "PCA-2D",
        "PCA-3D",
        "t-SNE-2D",
        "t-SNE-3D"]
    # imputers = st.sidebar.selectbox("Imutation algorithm", imputers_methods,0)
    clustering_methods = st.sidebar.selectbox(
        label="Choose...", options=clustering_methods
    )
    
    if clustering_methods == "t-SNE-2D" or clustering_methods == "t-SNE-3D" :
        n_iter = st.sidebar.slider('n_iter', 50, 1000, 300)
        perplexity = st.sidebar.slider('perplexity',5, 50, 40)
      
    btn1 = st.sidebar.button("Run-part-2")
    
    if btn1:
    
        X_std = StandardScaler().fit_transform(X_imputeds)
        if clustering_methods == "PCA-2D":
            fig, ax = plt.subplots()
            plt.show()
            pca = PCA(n_components=2)
            principalComponents = pca.fit_transform(X_std)
            PCA_components = pd.DataFrame(principalComponents)
            plt.scatter(PCA_components[0], PCA_components[1], alpha=.75, color='blue')
            plt.xlabel('PCA 1')
            plt.ylabel('PCA 2')
               
            st.pyplot(fig)
            plt.close()
            st.write("It's done by PCA-2D method!")
        
            #                 # X_imputed=gain(np.array(data2), gain_parameters)
        
        elif clustering_methods=="PCA-3D":
            fig, ax = plt.subplots()
            ax = fig.add_subplot(projection='3d')
        
            plt.show()
            pca = PCA(n_components=3)
            principalComponents = pca.fit_transform(X_std)
            PCA_components = pd.DataFrame(principalComponents)
            ax.scatter(PCA_components[0], PCA_components[1],PCA_components[2], alpha=.75, color='blue')
            ax.set_xlabel('PCA 1')
            ax.set_ylabel('PCA 2')
            ax.set_zlabel('PCA 3')
        
            st.pyplot(fig)
            plt.close()
            st.write("It's done by PCA-3D method!")
            
         
        elif clustering_methods=="t-SNE-2D": 
            fig, ax = plt.subplots()
            plt.show()
            tsne = TSNE(n_components=2, verbose=1, perplexity=perplexity, n_iter=n_iter)
            tsne_results = tsne.fit_transform(X_std)
            
            plt.scatter(tsne_results[:,0], tsne_results[:,1], alpha=.75, color='blue')
            plt.xlabel('t-SNE-1')
            plt.ylabel('t-SNE-2')
               
            st.pyplot(fig)
            plt.close()
            st.write("It's done by t-SNE-2D method!")
            
        else:
            
            fig, ax = plt.subplots()
            ax = fig.add_subplot(projection='3d')
            plt.show()
            tsne = TSNE(n_components=3, verbose=1, perplexity=perplexity, n_iter=n_iter)
            tsne_results = tsne.fit_transform(X_std)
            ax.scatter(tsne_results[:,0], tsne_results[:,1],tsne_results[:,2], alpha=.75, color='blue')
            ax.set_xlabel('t-SNE-1')
            ax.set_ylabel('t-SNE-2')
            ax.set_zlabel('t-SNE-3')
        
            st.pyplot(fig)
            plt.close()
            st.write("It's done by t-SNE-3D method!")
    
# ####################  show the importance features ##########################################
    st.header("The importance features")
    st.sidebar.header("The importance features")
    def get_param_lgbm1(key):
        min_child_weight1 = st.sidebar.slider('min_child_weight', 0.0, 1.0, .03,key=key)
        feature_fraction1 = st.sidebar.slider('feature_fraction', 0.0, 1.0, .03,key=key)
        learning_rate1 = st.sidebar.slider('learning_rate', 0.0, 1.0, 0.1,key=key)
        subsample1= st.sidebar.slider('subsample',0.0, 1.0, .6,key=key)
        min_data_in_leaf1 = st.sidebar.slider('min_data_in_leaf:' , 1,10,5,key=key)
        bagging_fraction1 = st.sidebar.slider('bagging_fraction', 0.0, 1.0, 0.4,key=key)
        reg_alpha1 = st.sidebar.slider('reg_alpha :', 0.0, 1.0, .3,key=key)
        reg_lambda1 = st.sidebar.slider('reg_lambda :', 0.0, 1.0, .6,key=key)
        n_estimators1 = st.sidebar.slider('n_estimators', 10, 1000, 100,key=key)

        params = {            'min_child_weight':min_child_weight1,
                              'feature_fraction': feature_fraction1,
                              'bagging_fraction': bagging_fraction1,
                              'min_data_in_leaf': min_data_in_leaf1,
                              'objective': 'regression',
                              'max_depth': -1,
                              'learning_rate':learning_rate1,
                              "boosting_type": "gbdt",
                              "bagging_seed": 11,
                              "metric": 'mse',
                              "verbosity": -1,
                              'reg_alpha':reg_alpha1,
                              'reg_lambda':reg_lambda1,
                              'subsample': subsample1,
                              'n_estimators' :n_estimators1,
                              }
        return params
    feature_imp =["LASSO", "LightGBM"]
    # imputers = st.sidebar.selectbox("Imutation algorithm", imputers_methods,0)
    feature_imp = st.sidebar.selectbox(label="Choose...", options=feature_imp)
    
    if feature_imp == "LightGBM":
        n_feat = st.sidebar.slider('choose the importance features to show',1, np.shape(data)[1], 5)
        params=get_param_lgbm1("a")
    elif feature_imp == "LASSO":
          n_feat = st.sidebar.slider('choose the importance features to show',1, np.shape(data)[1], 5)
    
        # imp_lgb = Misslgb(random_state=1337)
        # X_imputed0 = imp_lgb.fit_transform(data)
        # st.write("It's done by Mislighbm method!")
    @st.cache
    def get_feature_imp_LASSO(data,labels):
        
        labels=labels.ravel()
        labels=(labels-min(labels))/(max(labels)-min(labels))
        scaler = preprocessing.StandardScaler()
        scaler.fit(data)
        data = scaler.transform(data)
        # Create y from output
        importance = np.zeros((np.shape(data)[1],5))
        
        rkf = RepeatedKFold(n_splits=5, n_repeats=1, random_state=8725)
        itr=0
        for train, test in rkf.split(labels):
            X_train = data[train]
            y_train = labels[train] 
            # Lasso method (FEature selection and Prediction)
            clf = LassoCV(cv=5, random_state=0).fit(X_train, y_train)
            importance[:,itr] = clf.coef_
            itr +=1
        sum_importance=importance.sum(axis=1)/5
       
        return(sum_importance)
    
    @st.cache
    def get_feature_imp_Lgbm(data,labels,params):
        
        labels=labels.ravel()
        labels=(labels-min(labels))/(max(labels)-min(labels))
        if  len(np.unique(labels))==2:
            params['metric']='auc' 
            params['objective']='binary'
        # min_max_scaler = preprocessing.MinMaxScaler()
        # labels = min_max_scaler.fit_transform(np.array(labels))
        features={}
        # pre-prcoessing data
        scaler = preprocessing.StandardScaler()    
        scaler.fit(data)
        # transform the test test
        data = scaler.transform(data)
    
        rkf = RepeatedKFold(n_splits=5, n_repeats=1, random_state=8725)
        w=0    
        for train, test in rkf.split(labels):
            X_train = data[train]
            y_train = labels[train]
            
            X_train=pd.DataFrame(X_train)  
            #find the importance features 
            d_train = lgb.Dataset(X_train, y_train)
            clf = lgb.train(params, d_train, params.get('n_estimators'))
            features[w]=clf.feature_importance()
            w +=1
    
        features_avg_folds=pd.DataFrame.from_dict(features).mean(1)
        # features_avg_folds=(features[0]+features[1]+features[2]+features[3]+features[4])/5
        # feature_imp= pd.DataFrame({'Value':features_avg_folds,'Feature':feature_names})
        return features_avg_folds
    
    # labels=diabetes.target
    
    btn3 = st.sidebar.button("Run-part-3")
    
    if btn3:
        st.set_option('deprecation.showPyplotGlobalUse', False)
    
        if feature_imp=="LASSO":
            fig, ax = plt.subplots()
            out=get_feature_imp_LASSO(X_imputeds,labels)
            out1 = pd.DataFrame(out,colnamess)
            out1 = out1.reset_index()
            out1.columns =['Name','Value']
            out1= out1.sort_values("Value", key=abs, ascending = False)
            # out2=out1.iloc[0:parameters,:]
            # out2=out2.set_index('Name')
            # out2= out2.sort_values("Value", ascending = True)
            # # out1=abs(out1)
            plt.show()
            plt.figure(figsize=(10,5))
            sns.set(font_scale =1)
            sns.barplot(x="Value", y="Name", data=out1.sort_values(by="Value", key=abs,ascending=False)[0:n_feat])
            plt.title('Most Important Features')
            plt.tight_layout()
            plt.show()
    
            # fig = px.bar(out2, x='Value', y='Name')
            # st.plotly_chart(fig)
            st.pyplot()
            plt.close()
        elif feature_imp=="LightGBM":
            print(params)
            fig = plt.figure()
            # plt.show()
            out=get_feature_imp_Lgbm(X_imputeds,labels,params)
            out1 = pd.DataFrame(out)
            out1 = out1.reset_index()
            out1.columns =['Name','Value']
            out1.Name=colnamess
            plt.figure(figsize=(10,5))
            sns.set(font_scale =1)
            sns.barplot(x="Value", y="Name", data=out1.sort_values(by="Value", ascending=False)[0:n_feat])
            plt.tight_layout()
    
            plt.title('Most Important Features')
            
            st.pyplot()
            plt.close()
# # # #################### Classifier part ##########################################
    st.header("Classifier algorithm")
    st.sidebar.header("Classifier algorithm")
    from sklearn.feature_selection import SelectFromModel
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
    @st.cache
    def lgbmclassfier(data,labels,params):
        
        labels=labels.ravel()
        # labels=(labels-min(labels))/(max(labels)-min(labels))
    
        if len(np.unique(labels))==2:
            params['metric']='auc' 
            params['objective']='binary' 
        # pre-prcoessing data
        scaler = preprocessing.StandardScaler()
        scaler.fit(data)   
        # transform the test test
        data = scaler.transform(data)
        
        out_pre1 = np.zeros(len(labels))
        # apply 5-fold Cross validation method to split train-test data       
        rkf = RepeatedKFold(n_splits=5, n_repeats=1, random_state=8725)
        
        for train, test in rkf.split(labels):
            X_train = data[train]
            X_test = data[test]
            y_train = labels[train]
    
            X_train=pd.DataFrame(X_train)  
            X_test=pd.DataFrame(X_test)
            
            #find the importance features 
            d_train = lgb.Dataset(X_train, y_train)
            clf = lgb.train(params, d_train,params.get('n_estimators'))
            #select the importance features to bulid a predictive model
            idx=np.where(clf.feature_importance()>0)
            # train classifier with the best paramter set and he importance features
            d_train = lgb.Dataset(X_train.iloc[:,idx[0]], y_train)
            
            clf = lgb.train(params, d_train,params.get('n_estimators'))
    
            y_pred3=clf.predict(X_test.iloc[:,idx[0]])
    
            #caluclate the AUC for testing data prediction for each fold of each iteration
            out_pre1[test]=np.array(y_pred3)
            del clf  
               
        return   out_pre1
    @st.cache
    def lassoclassfier(data,labels):
    
        labels=labels.ravel()
        
        scaler = preprocessing.StandardScaler()
        scaler.fit(data)
        data = scaler.transform(data)
        
        # Create y from output
        out_pre_lasso = np.zeros(len(labels))
        
        rkf = RepeatedKFold(n_splits=5, n_repeats=1, random_state=8725)
        for train, test in rkf.split(labels):
            X_train = data[train]
            X_test = data[test]
            y_train = labels[train]
            
            # Lasso method (FEature selection and Prediction)
            clf = LassoCV(cv=5, random_state=0).fit(X_train, y_train)
            importance = np.abs(clf.coef_)
            idx=np.where(importance!=0)
            
            nsamples, nx, ny = X_train[:,idx].shape
            X_important_train = X_train[:,idx].reshape((nsamples,nx*ny))
            clf = LassoCV(cv=5, random_state=0).fit(X_important_train, y_train)
            
            
            nsamples, nx, ny = X_test[:,idx].shape
            X_important_test = X_test[:,idx].reshape((nsamples,nx*ny))
            y_pred_lasso=clf.predict(X_important_test)
            out_pre_lasso[test]=np.array(y_pred_lasso)
         
        return  out_pre_lasso
    @st.cache  
    def RFclassfier(data,labels,n_trees,n_feat,min_samples_split):
    
        labels=labels.ravel()
        
        scaler = preprocessing.StandardScaler()
        scaler.fit(data)
        data = scaler.transform(data)
        
        # Create y from output
        out_pre_RF = np.zeros(len(labels))
        
        rkf = RepeatedKFold(n_splits=5, n_repeats=1, random_state=8725)
        for train, test in rkf.split(labels):
            X_train = data[train]
            X_test = data[test]
            y_train = labels[train]
            # Random Forest method (FEature selection and Prediction)
            # Print the name and gini importance of each feature
            sfm = SelectFromModel(RandomForestRegressor(n_estimators=n_trees, random_state=0 ,n_jobs=-1), max_features=n_feat)
            # sfm = SelectFromModel(clf, threshold=0.003)
            
            # Train the selector
            sfm.fit(X_train, y_train)    
            
            X_important_train = sfm.transform(X_train)
            X_important_test = sfm.transform(X_test)
            
            # Create a new random forest classifier for the most important features
            clf_important = RandomForestRegressor(n_estimators=n_trees, min_samples_split=min_samples_split,random_state=0 ,n_jobs=-1)
            
            # Train the new classifier on the new dataset containing the most important features
            clf_important.fit(X_important_train, y_train)
            
            # Apply The Full Featured Classifier To The Test Data
            y_pred_RF = clf_important.predict(X_important_test)
            
            out_pre_RF[test]=np.array(y_pred_RF)
    
        return  out_pre_RF
    @st.cache
    def knnclassfier(data,labels,n_neighbors):
    
        labels=labels.ravel()    
        scaler = preprocessing.StandardScaler()
        scaler.fit(data)
        data = scaler.transform(data)
    
        # Create y from output
        out_pre_knn = np.zeros(len(labels))
        
        rkf = RepeatedKFold(n_splits=5, n_repeats=1, random_state=8725)
        for train, test in rkf.split(labels):
            X_train = data[train]
            X_test = data[test]
            y_train = labels[train]
            # KNN method 
            
            if len(np.unique(labels)) == 2:
                knn=KNeighborsClassifier(n_neighbors=n_neighbors)
            else:  
                knn=KNeighborsRegressor(n_neighbors=n_neighbors) 
    
            knn.fit(X_train,y_train)
            # Apply The Full Featured Classifier To The Test Data
            y_pred_knn =knn.predict(X_test)
            
            out_pre_knn[test]=np.array(y_pred_knn)
    
        return  out_pre_knn
    
    def get_all_imputed(data):
        X_imputed=[]
        imp_xgb = Missxgb.Missxgb(random_state=1337)
        X_imputed1=imp_xgb.fit_transform(data)
        get_imouted_data(X_imputed1,colnamess, "Misslgbm")
        X_imputed.append(X_imputed1)

        imp_lgb = misslgb.Misslgb(random_state=1337)
        X_imputed1=imp_lgb.fit_transform(data)
        get_imouted_data(X_imputed1,colnamess, "Misxgboost")
        X_imputed.append(X_imputed1)
        
        imp_forest = MissForest(random_state=1337)
        X_imputed1=imp_forest.fit_transform(data)
        get_imouted_data(X_imputed1,colnamess, "MissForest")
        X_imputed.append(X_imputed1)

        # imp_forest = MissForest(random_state=1337)
        imp_knn = KNNImputer(n_neighbors=5)
        X_imputed1=imp_knn.fit_transform(data)
        get_imouted_data(X_imputed1,colnamess, "KNNImputer")
        X_imputed.append(X_imputed1)
        
        X_imputed1=imp_median.fit_transform(data)
        get_imouted_data(X_imputed1,colnamess, "Median")
        X_imputed.append(X_imputed1)
        
        X_imputed1=imp_mean.fit_transform(data)
        get_imouted_data(X_imputed1,colnamess, "Mean")
        X_imputed.append(X_imputed1)
        # imp_median.fit_transform(data)
        
        X_imputed1=MICE().fit_transform(data)
        get_imouted_data(X_imputed1,colnamess, "MICE")
        X_imputed.append(X_imputed1)
        gain_parameters = {'batch_size': 32,
                      'hint_rate': .9,
                      'alpha': 100,
                      'iterations': 10000}
        X_imputed1=gain(np.array(data), gain_parameters)
        get_imouted_data(X_imputed1,colnamess, "GAIN")
        X_imputed.append(X_imputed1)
        
        X_imputed1=SoftImpute().fit_transform(data)
        get_imouted_data(X_imputed1,colnamess, "SoftImpute")
        X_imputed.append(X_imputed1)

        return X_imputed
    
    
    Classifier_methods =[
        "My data do not need to classifier!" ,  
        "RandomForest",
        "LASSO",
        "LightGBM",
        "KNN",
        "ALL"]
    # imputers = st.sidebar.selectbox("Imutation algorithm", imputers_methods,0)
    Classifier_methods = st.sidebar.selectbox(
        label="Choose...", options=Classifier_methods
    )
    
    if Classifier_methods == "RandomForest" :
        n_feat = st.sidebar.slider('choose the number of feature for features selction ',1, np.shape(data)[1], 5)
        n_trees = st.sidebar.slider('n_trees', 50, 5000, 100)
        min_samples_split = st.sidebar.slider('min_samples_split',2, 15, 3)
        
    elif Classifier_methods == "KNN" :    
          n_neighbors = st.sidebar.slider('n_neighbors', 2, 15, 3)
    
    elif Classifier_methods == "LightGBM" :    
        n_feat = st.sidebar.slider('choose the number of feature for features selction ',1, np.shape(data)[1], 5)
        params=get_param_lgbm1("b")
    
    elif Classifier_methods == "ALL" :    
          st.write("It just works for imputed data and evaluates them based on prediction resul!")
        
    
    btn4 = st.sidebar.button("Run-part-4")
    from sklearn import metrics
    if btn4:
        if Classifier_methods == "LightGBM" :  
            out = lgbmclassfier(X_imputeds,labels,params)
            if len(np.unique(labels)) == 2:
                result = np.sqrt(metrics.roc_auc_score(labels, out)) 
                st.write('the AUC value is : ' + str(result) )  
            else:  
                result = np.sqrt(metrics.mean_squared_error(labels, out)) 
                st.write('the MSE value is : ' + str(result) )  
              
        if Classifier_methods == "RandomForest":  
            out = RFclassfier(X_imputeds,labels,n_trees,n_feat,min_samples_split)
            if len(np.unique(labels)) == 2:
                result = np.sqrt(metrics.roc_auc_score(labels, out)) 
                st.write('the AUC value is : ' + str(result) )  
            else:  
                result = np.sqrt(metrics.mean_squared_error(labels, out)) 
                st.write('the MSE value is : ' + str(result) )  
           
        if Classifier_methods == "LASSO":  
            out =  lassoclassfier(X_imputeds,labels)
            if len(np.unique(labels)) == 2:
                result = np.sqrt(metrics.roc_auc_score(labels, out)) 
                st.write('the AUC value is : ' + str(result) )  
            else:  
                result = np.sqrt(metrics.mean_squared_error(labels, out)) 
                st.write('the MSE value is : ' + str(result) )  
              
        if Classifier_methods == "KNN" :  
            out = knnclassfier(X_imputeds,labels,n_neighbors)
            if len(np.unique(labels)) == 2:
                result = np.sqrt(metrics.roc_auc_score(labels, out)) 
                st.write('the AUC value is : ' + str(result) )  
            else:  
                result = np.sqrt(metrics.mean_squared_error(labels, out)) 
                st.write('the MSE value is : ' + str(result) )  
            
            
        if Classifier_methods == "ALL": 
            result=[]
            X_imputeds=get_all_imputed(data)
            if X_imputeds is not None:
                for i in range(0,9):
                    out =  lassoclassfier(X_imputeds[i],labels)
                    if len(np.unique(labels)) == 2:
                          result1 = metrics.roc_auc_score(labels, out)
                          result.append(result1)
                    else:  
                          result1 = np.sqrt(metrics.mean_squared_error(labels, out)) 
                          result.append(result1)
                    # 
            plt.figure()
            plt.show()
    
            ax = fig.add_axes([0,0,1,1])
            langs = ['Misslightgbm', 'Missxgboost','Missforest', 'KNNimputer','Median','Mean', 'MICE', 'GAIN','Softimpute']
            
            # result=[23,17,35,29,12,15,18,19]
            ax.set_ylabel('RMSE')
            if len(np.unique(labels)) == 2: ax.set_ylabel('AUC')
            # plt.xlabel('Diffrent imputed algprithms ')
            # plt.title('prediction after imputed data using diffrent imputed algprithm')
            h=np.round(10*np.min(result)-.5)/10
            print(result)
            result=np.array(result)-h
            result =result.tolist()
            print(result)
            ax.bar(langs,result,width = 0.4, bottom=h)
            ax.set_xticks(langs)    # This ensures we have one tick per year, otherwise we get fewer
            ax.set_xticklabels(langs, rotation='vertical')
            # ax.set_xlabel("Different imputed algorithms")
            ax.set_title('Prediction after imputed data applying different imputed algorithm')
            st.pyplot(fig)
            plt.close()
        

        
        
        
