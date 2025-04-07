# Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student

## AIM:
To write a program to implement the the Logistic Regression Model to Predict the Placement Status of Student.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
```
1. Import necessary libraries (pandas, LabelEncoder, train_test_split, etc.).
2. Load the dataset using pd.read_csv().
3. Create a copy of the dataset and drop unnecessary columns (sl_no, salary).
4. Check for missing and duplicate values using isnull().sum() and duplicated().sum().
5. Encode categorical variables using LabelEncoder() to convert them into numerical values.
```
## Program:
```
/*
Program to implement the the Logistic Regression Model to Predict the Placement Status of Student.
Developed by: V.Shriyha
RegisterNumber: 212224230267
*/
```
```
import pandas as pd
data=pd.read_csv("Placement_Data.csv")
data.head()
```
## Output:
![image](https://github.com/user-attachments/assets/e69df8cb-8b6e-4135-a511-4e71c1bc92b8)
```
data1=data.copy()
data1=data1.drop(["sl_no","salary"],axis=1)
data1.head()
```
## Output:
![image](https://github.com/user-attachments/assets/086b5cd8-a7bb-4d22-8bf2-e7d6c5680485)
```
data1.isnull().sum()
```
## Output:
![image](https://github.com/user-attachments/assets/14ce603b-211e-4e7c-886d-849e63a7aa6b)
```
data1.duplicated().sum()
```
## Output:
![image](https://github.com/user-attachments/assets/2cdc26b1-b80a-4a9a-bcfb-36e55b7b11fe)
```
from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
data1["gender"]=le.fit_transform(data1["gender"])
data1["ssc_b"]=le.fit_transform(data1["ssc_b"])
data1["hsc_b"]=le.fit_transform(data1["hsc_b"])
data1["hsc_s"]=le.fit_transform(data1["hsc_s"])
data1["degree_t"]=le.fit_transform(data1["degree_t"])
data1["workex"]=le.fit_transform(data1["workex"])
data1["specialisation"]=le.fit_transform(data1["specialisation"] )     
data1["status"]=le.fit_transform(data1["status"])       
data1
```
## Output:
![image](https://github.com/user-attachments/assets/dc265191-2ae1-4460-880a-e51b0862f7e4)
```
x=data1.iloc[:,:-1]
x
```
## Output:
![image](https://github.com/user-attachments/assets/07705888-0fcc-48c1-bbe1-d893ac726d05)
```
y=data1["status"]
y
```
## Output:
![image](https://github.com/user-attachments/assets/49f48b54-d0a7-4850-89d8-12dc52e1114d)
```
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)
from sklearn.linear_model import LogisticRegression
lr=LogisticRegression(solver="liblinear")
lr.fit(x_train,y_train)
y_pred=lr.predict(x_test)
y_pred
```
## Output:
![image](https://github.com/user-attachments/assets/c0785e7e-597f-45a5-8d0d-c16fdaf50ff7)
```
from sklearn.metrics import accuracy_score
accuracy=accuracy_score(y_test,y_pred)
accuracy
```
## Output:
![Screenshot 2025-04-07 155647](https://github.com/user-attachments/assets/1a2d772f-547c-48a6-a97a-d98b328878fb)
```
from sklearn.metrics import confusion_matrix
confusion=confusion_matrix(y_test,y_pred)
confusion
```
## Output:
![image](https://github.com/user-attachments/assets/1efe86d0-94e8-49a3-96e7-7f55f2649458)
```
from sklearn.metrics import classification_report
classification_report1 = classification_report(y_test,y_pred)
print(classification_report1)
lr.predict([[1,80,1,90,1,1,90,1,0,85,1,85]])
```
## Output:
![image](https://github.com/user-attachments/assets/2f9a3d00-2924-4ec4-80d9-f0f971341f8b)

## Result:
Thus the program to implement the the Logistic Regression Model to Predict the Placement Status of Student is written and verified using python programming.
