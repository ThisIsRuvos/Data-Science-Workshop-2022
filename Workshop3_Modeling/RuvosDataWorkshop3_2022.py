'''
2022 Summer Workshop Series
Modeling & Model Assessment
9/1/2022

Title: Part 3: A crash course in modeling

RUVOS
'''
#########################################################################
## Import Libraries
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, spearmanr
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB,BernoulliNB
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.ensemble import RandomForestClassifier,AdaBoostClassifier
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.metrics import classification_report,confusion_matrix

import warnings 
import seaborn as sns

#########################################################################
## Set working directory
os.chdir('C:/Users/INSERT YOUR DIRECTORY HERE')
print("Current Working Directory " , os.getcwd())

## Set matplotlib size dimensions
plt.rcParams["figure.figsize"] = (20,3)
#########################################################################
## Download Heart Disease Data Set
#https://www.kaggle.com/datasets/volodymyrgavrysh/heart-disease
#########################################################################
## Open Covid19 Data Set
df = pd.read_csv("heart.csv")
df.head()

plt.plot(df)
plt.legend(labels = features)

df.dtypes

## What is the heart disease split in the data set?
x = df['target'].value_counts()
plt.pie(x, labels=['Yes', 'No'], autopct="%1.1f%%", colors = ['#ff9999','#66b3ff'])
plt.title('Heart Disease')

#Notes
#What's missing? Nearly a 50/50 split of data between targets (this almost NEVER happens)
#Target is also binary, Yes/No. When the reality of heart disease is certainly shades of gray

## Nan/Null & Missing Assessment
print('Percent Missing: ')
for col in df.columns:
    pct_missing = np.mean(df[col].isnull())
    print('{} - {}%'.format(col, round(pct_missing*100)))


df_nan = df.loc[df.isnull().any(axis=1)]
df_nan.head()

## How is the split between Men and Women? 1 = Male, 0 = Female
ax = sns.countplot(x ='sex', hue = "target",data = df, palette=['#66b3ff', '#ff9999'])
plt.legend(title='Heart Disease', loc='upper left', labels=['No', 'Yes'])
for p in ax.patches:
    value=p.get_height() 
    if value <0:
        continue
    x = p.get_x()+.18
    y = p.get_y() + p.get_height() - 10
    ax.text((x), (y), int(value), fontsize=12,bbox=dict(facecolor='white', edgecolor='black', boxstyle='round', linewidth=0.65))

#Note: Interesting that we have more men without heart disease than men with, yet we have exceedingly more women with HD
#      than without. Anecdotally this feels wrong compared to disase in population.

## Correlation Matrix
plt.figure(figsize=(12,6))
sns.heatmap(df.corr(),annot=True)

## Take a closer look at Chest Pain
labels = ['Typical', 'Asymptotic', 'Nonanginal', 'Nontypical']
lbl_cnt = df['cp'].value_counts()
explode = (0, 0.1, 0, 0.1)
#add colors
colors = ['#ff9999','#66b3ff','#99ff99','#ffcc99']
fig1, ax1 = plt.subplots()
ax1.pie(lbl_cnt, explode=explode, labels=labels, colors=colors, autopct='%1.1f%%',
        shadow=True, startangle=90)
# Equal aspect ratio ensures that pie is drawn as a circle
ax1.axis('equal')
plt.tight_layout()
plt.show()

#########################################################################
## FEATURE ENGINEERING ##
# Ex:We have raw Cholesterol numbers (total serum). But suppose we wanted to convert that to a categorical to indicate
#    high cholesterol

## Table Below Taken from https://www.medicalnewstoday.com/articles/315900#recommended-levels #

## Version 1: Binary Flag

## Define a function to determine high cholesterol level
def high_cholesterol(row):
    #skipping age <=19 as none exist in this dataset
    if row['sex'] == 0 and row['chol'] > 200:
        return 1
    elif row['sex'] == 0 and row['chol'] > 200:
        return 0
    elif row['sex'] == 1 and row['chol'] > 200:
        return 1
    elif row['sex'] == 1 and row['chol'] > 200:
        return 0

## Pass function with apply to dataframe
df['high_ch'] = df.apply(lambda row: high_cholesterol(row), axis = 1)

print(df.head())

'''
Keep in mind there is a lot to decide when feature engineering:</b>
- Should this be a binary flag (is or is not high cholesterol) OR
- Do we create categorical levels (low risk, medium risk, high risk)...
- Do we then remove the numeric cholesterol variable in the model if we use the categorical high_ch?

Other Feature Engineering Examples
- Time series t-1 variables
- NLP text extraction
- variable combinations (x*y) (x+y) (x^2)
'''

## Feature Selection
## We already shared the correlation matrix which should give us a good idea already of what features are closely related
## Let's test as an example Cholesterol vs blood pressure (resting)

## Set high_ch flag to int
print('Values Range -1 to 1') # -1: Negatively Corr, 0: Uncorrellated, 1: Positively Correlated
## Pearson's Correlation (linear)
corr, _ = pearsonr(df['chol'], df['trestbps'])
print('Pearsons correlation Cholesterol vs resting BP: %.3f' % corr)

## Spearman's Correlation (non-linear)
corr, _ = spearmanr(df['chol'], df['trestbps'])
print('Spearmans correlation Cholesterol vs resting BP: %.3f' % corr)

## Using PCA to reduce total variables

features = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal']

# Separating out the features
x = df.loc[:, features].values

# Separating out the target
y = df.loc[:,['target']].values

# Standardizing the features (we'll discuss this later)
x = StandardScaler().fit_transform(x)

## Run PCA
pca = PCA(n_components=13)
principalComponents = pca.fit_transform(x)
principalDf = pd.DataFrame(data = principalComponents
             , columns = ['pca_1', 'pca_2', 'pca_3' ,'pca_4', 'pca_5', 'pca_6', 'pca_7', 'pca_8', 'pca_9', 
                          'pca_10', 'pca_11', 'pca_12', 'pca_13'])

## View new df with pca
pca_df = pd.concat([principalDf, df[['target']]], axis = 1)
pca_df.head()

## Visualize
plt.scatter(principalComponents[:, 0], principalComponents[:, 1],
            c=df.target, edgecolor='none', alpha=0.5, cmap=plt.cm.get_cmap('nipy_spectral', 10))
plt.xlabel('component 1')
plt.ylabel('component 2')
plt.colorbar()


## Natural Log Transformation
sns.histplot(df.age)

sns.histplot(np.log(df.age))

## Normalization (Min/Max Scaling)
## You can import libraries to scale for you, but let's do it manually here:
min_age = min(df.age)
max_age = max(df.age)

df['age_sc'] = ((df.age - min_age)) / ((max_age - min_age))
sns.histplot(df.age_sc)

## Another Glance at Correlation b/w variables and their distribution, to evaluate if we need tranformations
continuous_vars = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak']
sns.pairplot(df, hue='target', palette='rocket', vars = continuous_vars, kind = 'scatter', markers = ["o", "s"],
            corner = True, diag_kind = 'kde') #kernal density estimate

#########################################################################
## Train Validate Test
sc=StandardScaler()
X,y=df.drop(['target'],axis=1),df['target']

sc.fit(X)
df_sc=pd.DataFrame(sc.transform(X),columns=df.columns[:-1])
df_sc['target']=df['target']

#Show newly scaled variables 
df_sc.head()

## Split data into Train, Validate, and Test
X=df_sc.drop('target',axis=1)
y=df_sc['target']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.3)

#Print split dimensions
X_train.shape,y_train.shape,X_test.shape,y_test.shape

## Set models to compare
## Note: Models typically need to be 'tuned', here for demonstration purposes, we are using these models in default settings
classifiers=[LogisticRegression(),
             KNeighborsClassifier(),
             SVC(kernel='linear'),
             GaussianNB(),
             BernoulliNB(),
             RandomForestClassifier(),
             AdaBoostClassifier(),
             QuadraticDiscriminantAnalysis()]

## Create function to run classifiers
def predict(clf_list,score_list):
    for i in clf_list:
        i.fit(X_train,y_train)
        print('                ',i)
        print('Score =',i.score(X_test,y_test))
        print(confusion_matrix(y_test,i.predict(X_test)))
        print(classification_report(y_test,i.predict(X_test)))
        score_list.append(i.score(X_test,y_test))
        print('*'*80)
    return score_list

## Run classifiers
default_scores=[]
ds=predict(classifiers,default_scores)
ds

