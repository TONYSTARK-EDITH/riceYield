from django.shortcuts import render
from django.http import JsonResponse, Http404


import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
import matplotlib.pyplot as plt
from .models import *
import os.path as path
import joblib

def predict(n,p,k,rain):
    model = joblib.load("model.pkl")
    return model.predict([[n,p,k,rain]])


def dataSets():
    model = {}
    dataset = pd.read_csv('C:\\Users\\Tony Stark\\Desktop\\Mini\\data.csv')
    X = dataset.iloc[:, 1:-1].values
    y = dataset.iloc[:, -1].values
    y = y.reshape(-1, 1)
    st = []
    for i in dataset.iloc[:,1:].values:
        st.append(Mlmodel(nitrogen=i[0],phosphorus=i[1],pottasium=i[2],rainfall=i[3],rice_yield=i[4]))
    Mlmodel.objects.bulk_create(st)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
    svr_model = SVR(kernel='rbf')
    svr_model.fit(X_train, y_train)
    y_pred = svr_model.predict(X_test)
    st = [svr(predicted=i) for i in y_pred]
    svr.objects.bulk_create(st)
    s = r2_score(y_test, y_pred)
    model[s] = model.get(s, []) + [svr_model]
    rf_model = RandomForestRegressor(n_estimators=100, random_state=0)
    rf_model.fit(X_train, y_train)
    y_pred = rf_model.predict(X_test)
    st = [RF(predicted=i) for i in y_pred]
    RF.objects.bulk_create(st)
    s = r2_score(y_test, y_pred)
    model[s] = model.get(s, []) + [rf_model]
    dtr_model = DecisionTreeRegressor()
    dtr_model.fit(X_train, y_train)
    y_pred = dtr_model.predict(X_test)
    st = [DTR(predicted=i) for i in y_pred]
    DTR.objects.bulk_create(st)
    s = r2_score(y_test, y_pred)
    model[s] = model.get(s, []) + [dtr_model]

    model = dict(sorted(model.items(), key=lambda x: x[0], reverse=True))
    st = [RealValue(realVal=i) for i in y_test]
    RealValue.objects.bulk_create(st)

    t = list(model.values())[0][0]
    joblib.dump(t, 'model.pkl')


# Create your views here.
def home(request):
    if not path.exists("model.pkl"):
        dataSets()
    test,dtr,rf,sv = [],[],[],[]
    for j in RealValue.objects.all():
        test.append(str(j.realVal))
    for j in svr.objects.all():
        sv.append(str(j.predicted))
    for j in DTR.objects.all():
        dtr.append(str(j.predicted))
    for j in RF.objects.all():
        rf.append(str(j.predicted))
    test,sv,dtr,rf = ",".join(test),",".join(sv),",".join(dtr),",".join(rf)

    return render(request,'index.html',{'test':test,'svr':sv,'dtr':dtr,'rf':rf })


def Predict(request):
    if request.is_ajax():
        n,p,k,rain = float(request.POST.get('n'))/100,float(request.POST.get('p'))/100,float(request.POST.get('k'))/100,float(request.POST.get('rain'))
        val = predict(n, p, k, rain)[0]
        return JsonResponse({'val':val})
    else:
        return Http404("404 Not Found")