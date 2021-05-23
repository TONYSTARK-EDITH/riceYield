from django.shortcuts import render
from django.http import JsonResponse, Http404
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.preprocessing import StandardScaler
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
sc = StandardScaler()
scy = StandardScaler()
dataset = pd.read_csv('C:\\Users\\Tony Stark\\Desktop\\Mini\\data.csv')
X = dataset.iloc[:, 1:-1].values
y = dataset.iloc[:, -1].values
y = y.reshape(-1, 1)
X = sc.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
y_train = scy.fit_transform(y_train)
def predict(n,p,k,rain):
    model = joblib.load("model.pkl")
    return scy.inverse_transform(model.predict(sc.transform([[n,p,k,rain]])))


def dataSets():
    model = {}
    st = []
    for i in dataset.iloc[:,1:].values:
        st.append(Mlmodel(nitrogen=i[0],phosphorus=i[1],pottasium=i[2],rainfall=i[3],rice_yield=i[4]))
    Mlmodel.objects.bulk_create(st)
    svr_model = SVR(kernel='rbf')
    svr_model.fit(X_train, y_train)
    y_pred = scy.inverse_transform(svr_model.predict(X_test))
    s = r2_score(y_test, y_pred)
    st = [svr(predicted=i) for i in y_pred]
    svr.objects.bulk_create(st)
    model[s] = model.get(s, []) + [svr_model]
    rf_model = RandomForestRegressor(n_estimators=100, random_state=0)
    rf_model.fit(X_train, y_train)
    y_pred = scy.inverse_transform(rf_model.predict(X_test))
    s = r2_score(y_test, y_pred)
    st = [RF(predicted=i) for i in y_pred]
    RF.objects.bulk_create(st)
    model[s] = model.get(s, []) + [rf_model]
    dtr_model = DecisionTreeRegressor()
    dtr_model.fit(X_train, y_train)
    y_pred = scy.inverse_transform(dtr_model.predict(X_test))
    s = r2_score(y_test, y_pred)
    st = [DTR(predicted=i) for i in y_pred]
    DTR.objects.bulk_create(st)
    model[s] = model.get(s, []) + [dtr_model]
    lr_model = LinearRegression()
    lr_model.fit(X_train, y_train)
    y_pred = scy.inverse_transform(lr_model.predict(X_test))
    st = [mlr(predicted=i) for i in y_pred]
    mlr.objects.bulk_create(st)
    score = r2_score(y_test, y_pred)
    model[score] = model.get(score, []) + [lr_model]
    l = Ridge(alpha=0.2)
    l.fit(X_train, y_train)
    y_pred = scy.inverse_transform(l.predict(X_test))
    st = [ridge(predicted=i) for i in y_pred]
    ridge.objects.bulk_create(st)
    s = r2_score(y_test, y_pred)
    model[s] = model.get(s, []) + [l]

    l = Lasso(alpha=0.2)
    l.fit(X_train, y_train)
    y_pred = scy.inverse_transform(l.predict(X_test))
    st = [lasso(predicted=i) for i in y_pred]
    lasso.objects.bulk_create(st)
    s = r2_score(y_test, y_pred)
    model[s] = model.get(s, []) + [l]

    model = dict(sorted(model.items(), key=lambda x: x[0], reverse=True))
    st = [RealValue(realVal=i) for i in y_test]
    RealValue.objects.bulk_create(st)
    print(model)
    t = list(model.values())[0][0]
    print(t)
    joblib.dump(t, 'model.pkl')


# Create your views here.
def home(request):
    if not path.exists("model.pkl"):
        dataSets()
    test,dtr,rf,sv,Mlr,las,rid = [],[],[],[],[],[],[]
    for j in RealValue.objects.all():
        test.append(str(j.realVal))
    for j in svr.objects.all():
        sv.append(str(j.predicted))
    for j in DTR.objects.all():
        dtr.append(str(j.predicted))
    for j in RF.objects.all():
        rf.append(str(j.predicted))
    for j in mlr.objects.all():
        Mlr.append(str(j.predicted))
    for j in ridge.objects.all():
        rid.append(str(j.predicted))
    for j in lasso.objects.all():
        las.append(str(j.predicted))
    test,sv,dtr,rf,las,Mlr,rid = ",".join(test),",".join(sv),",".join(dtr),",".join(rf),",".join(las),",".join(Mlr),",".join(rid)

    return render(request,'index.html',{'test':test,'svr':sv,'dtr':dtr,'rf':rf,'las':las,'mlr':Mlr,'rid':rid })


def Predict(request):
    if request.is_ajax():
        n,p,k,rain = float(request.POST.get('n')),float(request.POST.get('p')),float(request.POST.get('k')),float(request.POST.get('rain'))
        val = predict(n, p, k, rain)[0]
        return JsonResponse({'val':val})
    else:
        raise Http404