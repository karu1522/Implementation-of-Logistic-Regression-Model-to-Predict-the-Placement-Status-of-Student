# Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student

## AIM:
To write a program to implement the the Logistic Regression Model to Predict the Placement Status of Student.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Clean and transform the raw data, then divide it into a training set (to build the model) and a testing set (to validate it).
2. Select your input features and establish parameters like the loss function and regularization to prevent overfitting.
3. Fit the model by adjusting parameters to minimize the loss function using the training data.
4. Assess performance on test data using metrics like Accuracy and F1 Score; if results are poor, refine the features or hyperparameters.
5. Apply the finalized model to new data to predict outcomes and analyze the coefficients to understand how each variable influences the final result.

## Program:
```
Program to implement the the Logistic Regression Model to Predict the Placement Status of Student.
Developed by: Karthic U
RegisterNumber: 212224040151
```

## Head values:
```py
import pandas as pd
df=pd.read_csv('Placement_Data.csv')
df.head()
```

## Output:
<img width="1397" height="247" alt="image" src="https://github.com/user-attachments/assets/ebe98176-b78c-4d4b-9182-820abb24e548" />


## Salary data:
```py
d1=df.copy()
d1=d1.drop(["sl_no","salary"],axis=1)
d1.head()
```

## Output:
<img width="1242" height="246" alt="image" src="https://github.com/user-attachments/assets/2ae12c64-66e9-4a73-ba6d-7f4d31cae489" />

## Checking Null function:
```py
d1.isnull().sum()
```

## Output:
<img width="246" height="609" alt="image" src="https://github.com/user-attachments/assets/6f47cb7d-ae74-4eb5-ae63-673d15fc34c0" />

## Duplicate data:
```py
d1.duplicated().sum()
```

## Output:
<img width="192" height="34" alt="image" src="https://github.com/user-attachments/assets/7f8be665-837e-4573-ab69-d8a818486dda" />


## Data status:
```py
from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
d1['gender']=le.fit_transform(d1["gender"])
d1["ssc_b"]=le.fit_transform(d1["ssc_b"])
d1["hsc_b"]=le.fit_transform(d1["hsc_b"])
d1["hsc_s"]=le.fit_transform(d1["hsc_s"])
d1["degree_t"]=le.fit_transform(d1["degree_t"])
d1["workex"]=le.fit_transform(d1["workex"])
d1["specialisation"]=le.fit_transform(d1["specialisation"])
d1["status"]=le.fit_transform(d1["status"])
d1
```

## Output:
<img width="1147" height="505" alt="image" src="https://github.com/user-attachments/assets/58f957d6-c9d5-4a33-89b9-a3aee499dcbc" />


## X data:
```py
x=d1.iloc[:, : -1]
x
```

## Output:
<img width="1069" height="505" alt="image" src="https://github.com/user-attachments/assets/c20b1514-a4a6-4e29-b7bf-2650190ff962" />


## Y data:
```py
y=d1["status"]
y
```

## Output:
<img width="228" height="555" alt="image" src="https://github.com/user-attachments/assets/3ddf7b16-0c19-4d7e-b385-65436e9e7b78" />


## Y prediction value:
```py
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.4,random_state=45)
from sklearn.linear_model import LogisticRegression
model=LogisticRegression(solver="liblinear")
model.fit(x_train,y_train)
y_pred=model.predict(x_test)
y_pred
```

## Output:
<img width="753" height="92" alt="image" src="https://github.com/user-attachments/assets/c69edcfa-96c9-4ec8-9281-404b732eb57c" />


## Accuracy value:
```py
from sklearn.metrics import accuracy_score,confusion_matrix,classification_report
accuracy=accuracy_score(y_test,y_pred)
accuracy
```

## Output:
<img width="237" height="33" alt="image" src="https://github.com/user-attachments/assets/3979d605-3a0d-415b-94a1-7cda05439dd7" />


## Confusion matrix:
```py
confusion=confusion_matrix(y_test,y_pred)
confusion
```

## Output:
<img width="223" height="52" alt="image" src="https://github.com/user-attachments/assets/44a0555a-0a79-4ef8-8051-e9b6b897e6c3" />

## Classification report:
```py
from sklearn.metrics import classification_report
classification_report=classification_report(y_test,y_pred)
print(classification_report)
```

## Output:
<img width="592" height="189" alt="image" src="https://github.com/user-attachments/assets/73e623fc-29bd-41c2-8fc1-470e6356ea04" />



## Result:
Thus the program to implement the the Logistic Regression Model to Predict the Placement Status of Student is written and verified using python programming.
