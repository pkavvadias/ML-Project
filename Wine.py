import pandas
from sklearn.model_selection import train_test_split 
from sklearn.svm import SVC
from sklearn.metrics import classification_report

#Read csv
data = pandas.read_csv('winequality-red.csv')
Y = data.quality
X = data.drop('quality', axis='columns')
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.25)

#Initialize SVC
svc = SVC()
svc.fit(X_train, Y_train)
predict = svc.predict(X_test)
print("Classification with unedited data:\n")
print(classification_report(Y_test, predict, zero_division=0))

#Remove 33% ph

#Gets ph column
data1 = X_train.pH
#Randomly selects 33% of ph values
data2 = data1.sample(frac=.33)
#Removes the selected ph values
data1 = data1.drop(data2.index)
#Gets a copy of the initial dataset without the ph column
data3 = X_train.drop('pH', axis='columns')
#Adds the ph column with the removed data
data4 = data3.join(data1)

#Prediction with empty ph column
X_train2 = data4.drop('pH', axis='columns')
X_test2 = X_test.drop('pH', axis='columns')
svc.fit(X_train2, Y_train)
predict = svc.predict(X_test2)
print("Classification with deleted pH column\n")
print(classification_report(Y_test, predict, zero_division=0))

#Fill empty values with mean of column
X_train3 = data4.fillna(X_train.pH.mean())
svc.fit(X_train3, Y_train)
predict = svc.predict(X_test)
print("Classification with average pH to the column\n")
print(classification_report(Y_test, predict, zero_division=0))
