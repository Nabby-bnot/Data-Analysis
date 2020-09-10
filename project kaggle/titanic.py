import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

data_train = pd.read_csv('train.csv')
data_test = pd.read_csv('test.csv')

dataset = data_train.copy(deep=True)
dataset = dataset.drop(['Cabin'],axis=1)


from sklearn.impute import SimpleImputer
imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
imputer = imputer.fit(dataset.iloc[:,5:6])
dataset.iloc[:,5:6] = imputer.transform(dataset.iloc[:,5:6])
dataset.isnull().sum()

from sklearn.ensemble import RandomForestClassifier

y = data_train["Survived"]

features = ["Pclass", "Sex", "SibSp", "Parch"]
X = pd.get_dummies(data_train[features])
X_test = pd.get_dummies(data_test[features])

model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=1)
model.fit(X, y)
predictions = model.predict(X_test)


output = pd.DataFrame({'PassengerId': data_test.PassengerId, 'Survived': predictions})
output.to_csv('my_submission.csv', index=False)


