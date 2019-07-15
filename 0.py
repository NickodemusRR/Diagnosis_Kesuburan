import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
import seaborn as sns

df = pd.read_csv('fertility.csv')
# print(df.head())
# print(df.describe())
# print(df.info())

# data preprocessing
from sklearn.preprocessing import LabelEncoder
label = LabelEncoder()

df ['Childish diseases'] = label.fit_transform(df['Childish diseases'])
# print(label.classes_)   # >> ['no' 'yes']

df ['Accident or serious trauma'] = label.fit_transform(df['Accident or serious trauma'])
# print(label.classes_)   # >> ['no' 'yes']

df ['Surgical intervention'] = label.fit_transform(df['Surgical intervention'])
# print(label.classes_)   # >> ['no' 'yes']

def fever(col):
    if col == 'no':
        return 0
    elif col == 'more than 3 months ago':
        return 1
    elif col == 'less than 3 months ago':
        return 2

# print(df['High fevers in the last year'].value_counts())
df['High fevers in the last year'] = df['High fevers in the last year'].apply(fever)
# print(df['High fevers in the last year'].value_counts())

# print(df['Frequency of alcohol consumption'].value_counts())
def alcohol(col):
    if col == 'hardly ever or never':
        return 0
    elif col == 'once a week':
        return 1
    elif col == 'several times a week':
        return 2
    elif col == 'every day':
        return 3
    elif col == 'several times a day':
        return 4

df['Frequency of alcohol consumption'] = df['Frequency of alcohol consumption'].apply(alcohol)
# print(df['Frequency of alcohol consumption'].value_counts())

# print(df['Smoking habit'].value_counts())
def smoking(col):
    if col == 'never':
        return 0
    elif col == 'occasional':
        return 1
    elif col == 'daily':
        return 2
df ['Smoking habit'] = df['Smoking habit'].apply(smoking)
# print(df['Smoking habit'].value_counts())


df['Diagnosis'] = df['Diagnosis'].apply(lambda x : 1 if x == 'Altered' else 0)

X = df.drop(['Season', 'Diagnosis'], axis=1)
y = df['Diagnosis']

# pd.set_option('display.max_columns', 500)
# print(X.head(2))
# >> Age, Childish diseases, Accident, Surgical, High fevers, Alcohol, Smoking, Sitting

# Because the number of datasets is little, I prefer to fit the model with all data
# Creating machine learning model
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

rf = RandomForestClassifier(n_estimators=100)
logreg = LogisticRegression(solver='liblinear')
svc = SVC(gamma='auto')

rf.fit(X, y)
logreg.fit(X, y)
svc.fit(X, y)

# print(rf.score(X,y))
# print(logreg.score(X,y))
# print(svc.score(X,y))

# Prediction
# >> Age, Childish diseases, Accident, Surgical, High fevers, Alcohol, Smoking, Sitting
Arin = [[29, 0, 0, 0, 0, 4, 2, 5]]
rfArin = rf.predict(Arin)[0]
logregArin = logreg.predict(Arin)[0]
svcArin = svc.predict(Arin)[0]

Bebi = [[31, 0, 1, 1, 0, 3, 0, 16]]
rfBebi = rf.predict(Bebi)[0]
logregBebi = logreg.predict(Bebi)[0]
svcBebi = svc.predict(Bebi)[0]

Caca = [[25, 1, 0, 0, 1, 0, 0, 7]]
rfCaca = rf.predict(Caca)[0]
logregCaca = logreg.predict(Caca)[0]
svcCaca = svc.predict(Caca)[0]

Dini = [[28, 0, 1, 1, 0, 0, 2, 16]]
rfDini = rf.predict(Dini)[0]
logregDini = logreg.predict(Dini)[0]
svcDini = svc.predict(Dini)[0]

Enno = [[42, 1, 0, 0, 0, 0, 0, 8]]
rfEnno = rf.predict(Enno)[0]
logregEnno = logreg.predict(Enno)[0]
svcEnno = svc.predict(Enno)[0]

predictions = pd.DataFrame([rfArin, rfBebi, rfCaca, rfDini, rfEnno], columns=['Random Forest'], index=['Arin', 'Bebi', 'Caca', 'Dini', 'Enno'])
predictions['Logistic Regression'] = [logregArin, logregBebi, logregCaca, logregDini, logregEnno]
predictions['SVC'] = [svcArin, svcBebi, svcCaca, svcDini, svcEnno]
predictions = predictions.replace({0:'Normal', 1:'Altered'})
print(predictions)