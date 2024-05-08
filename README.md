# Implementation-of-SVM-For-Spam-Mail-Detection

## AIM:
To write a program to implement the SVM For Spam Mail Detection.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
```
1.Import the required libraries.
2.Read the data frame using pandas.
3.Get the information regarding the null values present in the dataframe.
4.Split the data into training and testing sets.
5.Convert the text data into a numerical representation using CountVectorizer.
6.Use a Support Vector Machine (SVM) to train a model on the training data and make predictions on the testing data.
7.Finally, evaluate the accuracy of the model.
``` 

## Program:
```
/*
Program to implement the SVM For Spam Mail Detection..
Developed by: Sarish Varshan V
RegisterNumber: 212223230196 
*/
```
import chardet
file='spam.csv'
with open(file, 'rb') as rawdata:
result = chardet.detect(rawdata.read(100000))
result
import pandas as pd
data = pd.read_csv("")
data.head()
data.info()
data.isnull().sum()
X = data["v1"].values
Y = data["v2"].values
from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.2, random_state=0)
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer()
X_train = cv.fit_transform(X_train)
X_test = cv.transform(X_test)
from sklearn.svm import SVC
svc=SVC()
svc.fit(X_train,Y_train)
Y_pred = svc.predict(X_test)
print("Y_prediction Value: ",Y_pred)
from sklearn import metrics
accuracy=metrics.accuracy_score(Y_test,Y_pred)
accuracy
Program to implement the SVM For Spam Mail Detection.
```

## Output:
### result output:
![image](https://github.com/sarishvarshan/Implementation-of-SVM-For-Spam-Mail-Detection/assets/152167665/8b10d6d9-11b2-418a-808a-38ec6b123c0e)
### data.head()
![image](https://github.com/sarishvarshan/Implementation-of-SVM-For-Spam-Mail-Detection/assets/152167665/9bb9e702-02e8-48c5-947e-727951f99a7f)
### data.info()

![image](https://github.com/sarishvarshan/Implementation-of-SVM-For-Spam-Mail-Detection/assets/152167665/7af64dbc-cb29-4d01-864b-74d84ecfe8a0)
### data.isnull().sum()

![image](https://github.com/sarishvarshan/Implementation-of-SVM-For-Spam-Mail-Detection/assets/152167665/a2b78c10-f158-47e3-8040-39af2aca8549)
### Y_prediction Value

![image](https://github.com/sarishvarshan/Implementation-of-SVM-For-Spam-Mail-Detection/assets/152167665/7b0d2346-3832-48c2-ae4e-04cde7618da5)
### Accuracy Value

![image](https://github.com/sarishvarshan/Implementation-of-SVM-For-Spam-Mail-Detection/assets/152167665/3319270f-bda6-43a9-97c1-1a833ca95557)






## Result:
Thus the program to implement the SVM For Spam Mail Detection is written and verified using python programming.
