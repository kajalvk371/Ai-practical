from sklearn.ensemble import AdaBoostClassifier
from sklearn.svm import SVC
from sklearn import metrics
from sklearn.model_selection import train_test_split  
from sklearn.datasets import load_iris  

data = load_iris()  
X = data.data       
y = data.target     

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


svc = SVC(probability=True, kernel='linear')


abc = AdaBoostClassifier(n_estimators=50, estimator=svc, learning_rate=1)


model = abc.fit(X_train, y_train)


y_pred = model.predict(X_test)


print("Accuracy:", metrics.accuracy_score(y_test, y_pred))
