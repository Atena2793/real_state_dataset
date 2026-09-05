import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.metrics import r2_score, accuracy_score

df = pd.read_csv(r"A:\practice_files\real_estate_dataset.csv")
print(df.describe())
print(df.isnull().sum())
#print(df.duplicated().sum())

for i in df.columns:
    df[i] = df[i].fillna(df[i].median())
print(df.isnull().sum())

x = df.drop("Price", axis = 1)
y = df["Price"]

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size= 0.2, random_state=42)

reg = LinearRegression().fit(x_train, y_train)
knn = KNeighborsRegressor().fit(x_train, y_train)
svm = SVR().fit(x_train, y_train)

pred1 = reg.predict(x_test)
pred2 = knn.predict(x_test)
pred3 = svm.predict(x_test)

print("regrasion model is : %", r2_score(y_test, pred1)*100)
print("knn model is : %", r2_score(y_test, pred2)*100)
print("svm model is : %", r2_score(y_test, pred3)*100)