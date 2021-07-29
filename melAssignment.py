
import streamlit as st

import numpy as np
import pandas as pd
# import time
import matplotlib.pyplot as plt
import seaborn as sns
import sweetviz as sv


from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler

# from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder


st.set_page_config(layout="wide")

st.markdown("""
<style>
.big-font {
    font-size:50px !important;
}
</style>
""", unsafe_allow_html=True)

# st.title('##Hepatitis C Analysis')

st.markdown('<p class="big-font" colour=Blue,>Hepatitis C Analysis</p>', unsafe_allow_html=True)


st.markdown("""
<style>
.font {
    font-size:15px !important;
    color='skyblue;
}
</style>
""", unsafe_allow_html=True)

st.sidebar.markdown('<p class="font">A project by:</p>', unsafe_allow_html=True)

st.sidebar.write("<a href='https://www.linkedin.com/in/melissa-nathan-11168069/'>Melissa Nathan </a>", unsafe_allow_html=True)


st.markdown("""
<style>
.font1 {
    font-size:20px !important;
    color='green';
}
</style>
""", unsafe_allow_html=True)

st.markdown('<p "class=font1">Disclaimer :</p>', unsafe_allow_html=True)



st.markdown("""
<style>
.font2 {
    font-size:15px !important;
    color='green';
}
</style>
""", unsafe_allow_html=True)

st.markdown('<p class="font2">This is a web app using python libraries such as Streamlit, Sklearn etc, solely used for the purpose of learning</p>', unsafe_allow_html=True)



def load_data():
    data=pd.read_csv("HepatitisCdata.csv")
    return data

data=load_data()



st.sidebar.write('Before you continue, please read the [terms and conditions](https://www.gnu.org/licenses/gpl-3.0.en.html)')
show = st.sidebar.checkbox('I agree to the terms and conditions')
if show :
        st.write('Hepatitis C Data [(Original Data)](https://www.kaggle.com/fedesoriano/hepatitis-c-dataset)')
#         st.write('For further info on the dataset,please click this [link](https://www.kaggle.com/fedesoriano/hepatitis-c-dataset))
        st.write(data)
#         chart_data = pd.DataFrame(
#         np.random.randn(589, 2),
#         columns=['a', 'b', 'c'])

#         st.line_chart(chart_data)
       
else:
        st.write('Thank you, please read the terms and conditions to proceed')


# analyze = sv.analyze(data)
# st.write(analyze)

# def plotCorrelationMatrix(data, graphWidth):
#     filename = data
#     # filename = data.DataFrame
#     df = data.dropna() # drop columns & rows with NaN
#     df = data[[col for col in data if data[col].nunique() > 1]] # keep columns where there are more than 1 unique values
#     if data.shape[1] < 2:
#         print(f'No correlation plots shown: The number of non-NaN or constant columns ({data.shape[1]}) is less than 2')
#         return
#     corr = df.corr()
#     plt.figure(num=None, figsize=(graphWidth, graphWidth), dpi=80, facecolor='w', edgecolor='k')
#     corrMat = plt.matshow(corr, fignum = 1)
#     plt.xticks(range(len(corr.columns)), corr.columns, rotation=90)
#     plt.yticks(range(len(corr.columns)), corr.columns)
#     plt.gca().xaxis.tick_bottom()
#     plt.colorbar(corrMat)
#     plt.title(f'Correlation Matrix for {filename}', fontsize=15)
#     plt.show()

# st.write.plotCorrelationMatrix(data, 14)

# y = data['Category']
# y = LabelEncoder().fit_transform(y)

classifier_name = st.sidebar.selectbox(
    'Select classifier',
    ('KNN', 'SVM', 'Random Forest')
)

st.sidebar.write('To clean & process the data')
# button = st.sidebar.button('Click to clean & process the data')
show = st.sidebar.button('Click Here')
# st.sidebar.button('Click here')
# show = st.sidebar.button('To clean & process the data')
# show = st.sidebar.button("<font color='Aquamarine'>"Click to clean & process the data"</font>", unsafe_allow_html=True)
if show:
        data=pd.read_csv("HepatitisCdata.csv") 
        data = data.dropna()
        a = data.round(2)
        data = a.drop(['Unnamed: 0'], axis=1)
        data['Category'].loc[data['Category'].isin(["1=Hepatitis","2=Fibrosis", "3=Cirrhosis"])] = 1
        data['Category'].loc[data['Category'].isin(["0=Blood Donor", "0s=suspect Blood Donor"])] = 0
        data['Sex'].loc[data['Sex']=='m']=1
        data['Sex'].loc[data['Sex']=='f']=0
  
        st.write('Hepatitis C Data (Cleaned Data) ')
        st.write(data)
else: 
    st.write('')


fig = plt.figure(data)
plt.scatter(X,
               c=y, alpha=0.8,
               cmap='viridis')

plt.xlabel('Attributes')
plt.ylabel('Category')
plt.colorbar()
st.pyplot(fig)
    
X = data.drop(['Category'], axis=1)
y = data['Category']
y = LabelEncoder().fit_transform(y)


from sklearn.metrics import classification_report
from sklearn.neighbors import KNeighborsClassifier


test_data_ratio = st.sidebar.slider('Select testing size or ratio', 
                                    min_value= 0.01, 
                                    max_value = 0.5,
                                    value=0.2)
random_state = st.sidebar.slider('Select random state', 1, 589 ,value=10)

st.write("## 1: Summary (X variables)")
if len(X)==0:
   st.write("<font color='Aquamarine'>Note: Predictors @ X variables have not been selected.</font>", unsafe_allow_html=True)
else:
   st.write('Shape of predictors @ X variables :', X.shape)
   st.write('Summary of predictors @ X variables:', pd.DataFrame(X).describe())

st.write("## 2: Summary (y variable)")

if len(y)==0:
   st.write("<font color='Aquamarine'>Note: Label @ y variable has not been selected.</font>", unsafe_allow_html=True)
elif len(np.unique(y)) <6:
     st.write('Number of classes:', len(np.unique(y)))
     st.write('Shape of predictors @ y variables :', y.shape)
     st.write('Summary of predictors @ y variables:', pd.DataFrame(y).describe())

else: 
   st.write("<font color='red'>Warning: System detects an unusual number of unique classes. Please make sure that the label @ y is a categorical variable. Ignore this warning message if you are sure that the y is a categorical variable.</font>", unsafe_allow_html=True)
   st.write('Number of classes:', len(np.unique(y)))

def add_parameter_ui(clf_name):
    params = dict()
    if clf_name == 'SVM':
        C = st.sidebar.slider('C', 0.01, 10.0,value=1.0)
        # C = st.sidebar.write('C=10')
        params['C'] = C
    elif clf_name == 'KNN':
        K = st.sidebar.slider('K', 1, 15,value=5)
        # K = st.sidebar.write('K=5')
        params['K'] = K
    else:
        max_depth = st.sidebar.slider('max_depth', 2, 15,value=5)
        # max_depth = st.sidebar.write('max_depth=5')
        params['max_depth'] = max_depth
        n_estimators = st.sidebar.slider('n_estimators', 1, 100,value=10)
        # n_estimators = st.sidebar.write('n_estimatorse=10')
        params['n_estimators'] = n_estimators
    return params

params = add_parameter_ui(classifier_name)

def get_classifier(clf_name, params):
    clf = None
    if clf_name == 'SVM':
        clf = SVC(C=params['C'])
    elif clf_name == 'KNN':
        clf = KNeighborsClassifier(n_neighbors=params['K'])
    else:
        clf = clf = RandomForestClassifier(n_estimators=params['n_estimators'], 
            max_depth=params['max_depth'], random_state=random_state)
    return clf

clf = get_classifier(classifier_name, params)

st.write("## 3: Classification Report")
if len(X)!=0 and len(y)!=0: 

  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_data_ratio, random_state=random_state)
  labelencoder = LabelEncoder() 
  data['Category'] = labelencoder.fit_transform(data['Category'])
  scaler = StandardScaler()
  X_train_scaled = scaler.fit_transform(X_train)
  X_test_scaled = scaler.transform(X_test)    

  clf.fit(X_train_scaled, y_train)
  y_pred = clf.predict(X_test_scaled)

  st.write('Classifier:',classifier_name)
  st.write('Classification report:')
  report = classification_report(y_test, y_pred,output_dict=True)
  data = pd.DataFrame(report).transpose()
  # st.write(classification_report(ytest, ypred))
  st.write(data)

else: 
   st.write("<font color='Aquamarine'>Note: No classification report generated.</font>", unsafe_allow_html=True)


hide_streamlit_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            </style>
            """

import matplotlib.pyplot as plt

RandomForest.feature_importances_
list(X.columns)
important_factors = pd.DataFrame({'Factor': list(X.columns), 'Importance': RandomForest.feature_importances_})
# important_factors
important_factors.sort_values(by=['Importance'], ascending=False,inplace=True)
st.write(important_factors)




