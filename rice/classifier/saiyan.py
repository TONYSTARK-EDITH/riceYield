import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor

model = {}


def dataSets():
    dataset = pd.read_csv('C:\\Users\\Tony Stark\\Desktop\\Mini\\data.csv')
    X = dataset.iloc[:, :-1].values
    y = dataset.iloc[:, -1].values
    y = y.reshape(len(y), 1)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
    svr_model = SVR(kernel='rbf')
    svr_model.fit(X_train, y_train)
    y_pred = svr_model.predict(X_test)
    s = r2_score(y_test, y_pred)
    model[s] = model.get(s, []) + [svr_model]
    rf_model = RandomForestRegressor(n_estimators=100, random_state=0)
    rf_model.fit(X_train, y_train)
    y_pred = rf_model.predict(X_test)
    s = r2_score(y_test, y_pred)
    model[s] = model.get(s, []) + [rf_model]
    # poly_reg = PolynomialFeatures(degree=2)
    # X_poly = poly_reg.fit_transform(X_train)
    # mlr_model = LinearRegression()
    # mlr_model.fit(X_poly, y_train)
    # y_pred = mlr_model.predict(X_test)
    # s = r2_score(y_test, y_pred)
    # model[s] = model.get(s, []) + [mlr_model]
    dtr_model = DecisionTreeRegressor()
    dtr_model.fit(X_train, y_train)
    y_pred = dtr_model.predict(X_test)
    s = r2_score(y_test, y_pred)
    model[s] = model.get(s, []) + [dtr_model]
    print(model)



dataSets()
