import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from functools import reduce
import warnings
warnings.filterwarnings('ignore')
st.set_option('deprecation.showPyplotGlobalUse', False)
st.sidebar.title("Accuracy Scores")
st.title('Student Recommendation Dashboard')
st.subheader('''With online classes now becoming more prominent than ever teachers will or have a hard time evaluating and understanding their students thereby focus on those students who more need attention and direction.''')
st.subheader('''This project tries to explore ways to automate the process of assessing previous exam performance and predict if a student will drop the course or needs special attention or not.''')
assessments = pd.read_csv(r'assessments.csv')
studentAssessment = pd.read_csv(r'studentAssessment.csv')
studentInfo = pd.read_csv(r'studentInfo.csv')
studentRegistration = pd.read_csv(r'studentRegistration.csv')
studentVle = pd.read_csv(r'studentVle.csv', nrows=850000)
vle = pd.read_csv(r'vle.csv')
courses= pd.read_csv(r'courses.csv')
set1 = list(assessments.columns.values)
set2 = list(courses.columns.values)
set3 = list(studentAssessment.columns.values)
set4 = list(studentInfo.columns.values)
set5 = list(studentRegistration.columns.values)
set6 = list(studentVle.columns.values)
set7 = list(vle.columns.values)

all_columns = [set1, set2, set3, set4, set5, set6, set7]
columns_count = [assessments.shape,courses.shape,studentAssessment.shape, studentInfo.shape, studentRegistration.shape, studentVle.shape, vle.shape]
columns_header = ['assessments', 'courses', 'studentAssessment', 'studentInfo', 'studentRegistration', 'studentVle', 'vle' ]

#creating the data frame 
d={'Table Name':columns_header,'Rows, Columns': columns_count,'Column_names':all_columns}
df= pd.set_option('max_colwidth', 200)
df=pd.DataFrame(d)
st.subheader('Data Set Information')

# dropping null values
assessments.dropna(inplace=True)
courses.dropna(inplace=True)
studentAssessment.dropna(inplace=True)
studentInfo.dropna(inplace=True)
studentRegistration.dropna(inplace=True)
studentVle.dropna(inplace=True)
vle.dropna(inplace=True)

columns_count = [assessments.shape,courses.shape,studentAssessment.shape, studentInfo.shape, studentRegistration.shape, studentVle.shape, vle.shape]
d = {'Table Name':columns_header,'Rows, Columns': columns_count,'Column Names':all_columns}
df = pd.set_option('max_colwidth', 200)
df = pd.DataFrame(d)
df

#analysing the data based on gender,disability,age,highest education,education,region,imd
#result based on gender
gender = studentInfo.groupby(['gender'],as_index = False)
gender_count = gender['id_student'].count()
result_gender = studentInfo.groupby(['gender', 'final_result'],as_index = False)
result_gender_count = result_gender['id_student'].count()

merge= pd.merge(gender_count,result_gender_count, on='gender', how='left')
merge['i']=round((merge['id_student_y']/merge['id_student_x']),2)

merge=merge[['gender','final_result','i']]

female=merge.loc[merge['gender']=='F']
male=merge.loc[merge['gender']=='M']

fig=plt.figure()

ax=fig.add_subplot(111)


female.set_index('final_result',drop=True,inplace=True)
male.set_index('final_result',drop=True,inplace=True)
female.plot(kind='bar', ax=ax, width= 0.3, position=0)
male.plot(kind='bar', color='#2ca02c', ax=ax, width= 0.3, position=1)
plt.xlabel('Results')
plt.ylabel('result_status')

plt.title('Relation between perfomance of students to their gender')
plt.legend(['Male','Female'])
plt.show()
st.subheader('Result Based on gender')
st.pyplot()
st.set_option('deprecation.showPyplotGlobalUse', False)

#results based on age
age = studentInfo.groupby(['age_band'],as_index = False)
age_count = age['id_student'].count()
result_age = studentInfo.groupby(['age_band','final_result'],as_index = False)
result_age_count=result_age['id_student'].count()

merge = pd.merge(age_count, result_age_count, on ='age_band', how = 'left')
merge['_']=round((merge['id_student_y']/merge['id_student_x']),2)
merge = merge[['age_band','final_result','_']]

merge.set_index(['age_band','final_result']).unstack().plot(kind = 'barh' , stacked = True)

box =  ax.get_position()
ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])

plt.ylabel('Age')
plt.xlabel('Result')
plt.title('Age vs Result')
plt.legend(['Distinction' , 'Fail', 'Pass','Withdrawn'],loc='center left', bbox_to_anchor=(1,0.80))
plt.show()
st.subheader('Results based on age')
st.pyplot()




#Performance based on region


region = studentInfo.groupby(['region'],as_index = False)
region_count = region['id_student'].count()
result_region = studentInfo.groupby(['region', 'final_result'],as_index = False)
result_region_count = result_region['id_student'].count()

merge = pd.merge(region_count, result_region_count , on = 'region', how = 'left')
merge['_'] = round((merge['id_student_y']/merge['id_student_x']), 2)
merge = merge[['region','final_result', '_']]

merge.set_index(['region','final_result']).unstack().plot(kind="barh", stacked=True)

box = ax.get_position()
ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])

plt.ylabel('Region')
plt.xlabel('Result')
plt.title('Region vs Result')
plt.legend(['Distinction','Fail', 'Pass', 'Withdrawn'], loc='center left', bbox_to_anchor=(1, 0.85))
plt.show()
st.subheader('Region-wise Performance')
st.pyplot()

#prediction model
dfs=[studentAssessment, studentInfo, studentRegistration]
df_final= reduce(lambda left, right: pd.merge(left, right, on= 'id_student'), dfs)
df_final['final_result'].value_counts()

df_final = df_final.drop(['date_unregistration'],axis =1)

#converting final table into catagorical data
from sklearn import preprocessing
le =  preprocessing.LabelEncoder()
df_final = df_final.apply(le.fit_transform)

# Decision Tree
from sklearn.model_selection import train_test_split
from sklearn import tree
from sklearn import metrics

X = df_final.loc[:, df_final.columns != 'final_result']
y = df_final['final_result']
xTrain, xTest, yTrain, yTest = train_test_split(X, y,train_size = 0.75)

dt = tree.DecisionTreeClassifier(criterion='gini')
dt = dt.fit(xTrain, yTrain)
train_pred = dt.predict(xTrain)
test_pred = dt.predict(xTest)
print("Accuracy:{0:.3f}".format(metrics.accuracy_score(yTest, test_pred)),"\n")

# Gradient Boosting Regression
from sklearn.ensemble import GradientBoostingRegressor

X = df_final.loc[:, df_final.columns != 'final_result']
y = df_final['final_result']
xTrain, xTest, yTrain, yTest = train_test_split(X, y,train_size = 0.75)

gb = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=10, random_state=0, loss='huber')
gb = gb.fit(xTrain, yTrain)

print("Accuracy:{0:.3f}".format(gb.score(xTest, yTest)))

# Random Forest
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics

X = df_final.loc[:, df_final.columns != 'final_result']
y = df_final['final_result']
xTrain, xTest, yTrain, yTest = train_test_split(X, y,train_size = 0.75)

rf = RandomForestClassifier(n_estimators=10,random_state=33)
rf = rf.fit(xTrain, yTrain)
train_pred = rf.predict(xTrain)
test_pred = rf.predict(xTest)
print("Accuracy:{0:.3f}".format(metrics.accuracy_score(yTest, test_pred)),"\n")

st.markdown(' Note: We used decision trees to predict the student performance and the accuracy was not good so we used Random forest and gradient boosting techniques to improve the accuracy.')
st.markdown(' Here are the heatmaps for the accuracy scores for the techniques we have used')
# Function to plot accuracy

from sklearn.model_selection import learning_curve
import numpy as np

def plot_accuracy(model):

    train_sizes, train_scores, test_scores = learning_curve(model, xTrain, yTrain, n_jobs=-1, train_sizes=np.linspace(.1, 1.0, 5), verbose=0)

    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)
    test_mean = np.mean(test_scores, axis=1)
    test_std = np.std(test_scores, axis=1)
    
    plt.figure()
    plt.gca().invert_yaxis()
    plt.grid()
    plt.ylim(0.0, 1.1)
    plt.title("Accuracy Plot")
    plt.xlabel("Testing")
    plt.ylabel("Accuracy %")

    plt.fill_between(train_sizes, test_mean - test_std, test_mean + test_std, alpha=0.1)
        
    plt.plot(train_sizes, test_mean, 'bo-', color = "r", label="Test Score")
    
    #Decision Tree
    st.subheader('Accuracy for Decision Tree')
    plot_accuracy(dt)
    st.pyplot()
   
    #Decision Tree

import scikitplot as skplt
from sklearn.model_selection import cross_val_predict
if st.sidebar.button(' Confusion Matrix for decision tree'):
    st.header(' Confusion Matrix for decision tree')
    y_pred = cross_val_predict(dt, xTest, yTest)
    skplt.metrics.plot_confusion_matrix(yTest, y_pred, normalize=True)
   
    plt.show()
st.pyplot()

#Gradient Boosting
import numpy as np
import scikitplot as skplt
from sklearn.model_selection import cross_val_predict
if st.sidebar.button(' Confusion Matrix for gradient boosting'):
    st.header(' Confusion Matrix for gradient boosting') 
    y_pred = cross_val_predict(gb, xTest, yTest)
    y_pred = np.absolute(y_pred)

    skplt.metrics.plot_confusion_matrix(yTest, y_pred.round(), normalize=True)

    plt.show()
st.pyplot()

#Random Forest
import scikitplot as skplt
from sklearn.model_selection import cross_val_predict

if st.sidebar.button(' Confusion Matrix for random forest '):
     st.header(' Confusion Matrix for random forest') 
     y_pred = cross_val_predict(rf, xTest, yTest)
     skplt.metrics.plot_confusion_matrix(yTest, y_pred, normalize=True)

     plt.show()
st.pyplot()

