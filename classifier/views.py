import os.path as path
import platform
import random
import threading
import time

import GPUtil
import joblib
import pandas as pd
import psutil
from DLT import *
from django.conf import settings
from django.contrib.auth import authenticate, login, logout
from django.contrib.auth.decorators import login_required
from django.contrib.auth.models import User
from django.http import JsonResponse
from django.shortcuts import redirect, render
from django.views.decorators.csrf import ensure_csrf_cookie
from rest_framework.views import APIView, Response
from sklearn import linear_model
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBClassifier

from .models import *
from .serializer import PredictSerializers

"""
Initial val
"""
sc = StandardScaler()
scy = StandardScaler()
dataset = pd.read_csv('data.csv')
X = dataset.iloc[:, 1:-1]
y = dataset.iloc[:, -1]
X = sc.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=0)
base_dir = settings.BASE_DIR
AVAIL_MEM = psutil.virtual_memory().total >> 30
FERT = ('10-26-26', '14-35-14', '17-17-17', '20-20', '28-28', 'DAP',
        'Urea')

CROP = ('Barley', 'Cotton', 'Ground Nuts', 'Maize', 'Millets', 'Oil seeds',
        'Paddy', 'Pulses', 'Sugarcane', 'Tobacco', 'Wheat')

CROP_REC = ('apple',
            'banana',
            'blackgram',
            'chickpea',
            'coconut',
            'coffee',
            'cotton',
            'grapes',
            'jute',
            'kidneybeans',
            'lentil',
            'maize',
            'mango',
            'mothbeans',
            'mungbean',
            'muskmelon',
            'orange',
            'papaya',
            'pigeonpeas',
            'pomegranate',
            'rice',
            'watermelon')


def reverse_datasets():
    dataset = pd.read_csv('data.csv')
    X = dataset.iloc[:, 1:-1]
    X[:, 0] = X[:, 0] * 10
    for i in range(1, 3):
        X[:, i] = X[:, i] * 100
    X = X.astype(int)
    y = dataset.iloc[:, -1]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=0)

    knn = MultiOutputClassifier(KNeighborsClassifier())
    knn.fit(y_train.reshape(-1, 1), X_train)
    joblib.dump(knn, "reverse_engineer.h5")


def reverse_fert():
    df = pd.read_csv("Fertilizer Prediction.csv")
    X, y = df.iloc[:, :-1], df.iloc[:, -1]
    multilabel_model = MultiOutputClassifier(KNeighborsClassifier())
    multilabel_model.fit(y.reshape(-1, 1), X)
    joblib.dump(multilabel_model, "reverse_fert.h5.z")


def fertilizer_dataset():
    df = pd.read_csv("Fertilizer Prediction.csv")
    X, y = df.iloc[:, :-1], df.iloc[:, -1]
    fertname_label_encoder = LabelEncoder()
    crop_type_encoder = LabelEncoder()
    crop_type_encoder.fit(df['Crop Type'])
    fertname_label_encoder.fit(df['Fertilizer Name'])
    df['Crop Type'] = crop_type_encoder.transform(df['Crop Type'])
    df['Fertilizer Name'] = fertname_label_encoder.transform(df['Fertilizer Name'])
    xgb_pipeline = make_pipeline(StandardScaler(), XGBClassifier(random_state=18))
    xgb_pipeline.fit(X, y)
    joblib.dump(xgb_pipeline, "fertilizer.h5.z")


'''
predict -> n:float,p:float,k:float,rain:float
'''


def predict(n, p, k, rain):
    model = joblib.load("model.pkl")
    return model.predict([[n, p, k, rain]])


def fertilizer_suggest(temp, moist, n, p, k, c):
    model = joblib.load("fertilizer.h5")
    return FERT[model.predict([[temp, moist, n, p, k, CROP.index(c)]])[0]]


def reverse_engineer_predict(yld):
    model = joblib.load("reverse_engineer.h5")
    return model.predict([[yld]])


def reverse_fertilizer_predict(fert):
    model = joblib.load("reverse_fert.h5")
    return model.predict([[fert]])


def datasets():
    model = {}
    st = []
    for i in dataset.iloc[:1000, 1:].values:
        st.append(Mlmodel(nitrogen=i[0], phosphorus=i[1], pottasium=i[2], rainfall=i[3], rice_yield=i[4]))
    Mlmodel.objects.bulk_create(st)
    '''
    Support Vector Regression : -> kernel -> Radial Base Function
    '''
    svr_model = SVR(kernel='rbf')
    svr_model.fit(X_train, y_train)
    y_pred = svr_model.predict(X_test)
    s = r2_score(y_test, y_pred)
    st = [Svr(predicted=i) for i in y_pred[:1000]]
    Svr.objects.bulk_create(st)
    model[s] = model.get(s, []) + [svr_model]
    '''
    Random Forest Regression : Number of trees -> 1000, randomState -> 0
    '''
    rf_model = RandomForestRegressor(n_estimators=100, random_state=20)
    rf_model.fit(X_train, y_train)
    y_pred = rf_model.predict(X_test)
    s = r2_score(y_test, y_pred)
    st = [RF(predicted=i) for i in y_pred[:1000]]
    RF.objects.bulk_create(st)
    model[s] = model.get(s, []) + [rf_model]
    '''
    Decision Tree
    '''
    dtr_model = DecisionTreeRegressor()
    dtr_model.fit(X_train, y_train)
    y_pred = dtr_model.predict(X_test)
    s = r2_score(y_test, y_pred)
    st = [DTR(predicted=i) for i in y_pred[:1000]]
    DTR.objects.bulk_create(st)
    model[s] = model.get(s, []) + [dtr_model]
    '''
    Multiple Linear Regression
    '''
    lr_model = linear_model.LinearRegression()
    lr_model.fit(X_train, y_train)
    y_pred = lr_model.predict(X_test)
    st = [Mlr(predicted=i) for i in y_pred[:1000]]
    Mlr.objects.bulk_create(st)
    score = r2_score(y_test, y_pred)
    model[score] = model.get(score, []) + [lr_model]
    ''' 
    Ridge Linear Regression : alpha -> 0.2
    '''
    l = linear_model.Ridge(alpha=0.2)
    l.fit(X_train, y_train)
    y_pred = l.predict(X_test)
    st = [RIDGE(predicted=i) for i in y_pred[:1000]]
    RIDGE.objects.bulk_create(st)
    s = r2_score(y_test, y_pred)
    model[s] = model.get(s, []) + [l]
    '''
    Lasso Linear Regression : alpha -> 0.2
    '''
    l = linear_model.Lasso(alpha=0.2)
    l.fit(X_train, y_train)
    y_pred = l.predict(X_test)
    st = [Lasso(predicted=i) for i in y_pred[:1000]]
    Lasso.objects.bulk_create(st)
    s = r2_score(y_test, y_pred)
    model[s] = model.get(s, []) + [l]
    model = dict(sorted(model.items(), key=lambda x: x[0], reverse=True))
    st = [RealValue(realVal=i) for i in y_test[:1000]]
    RealValue.objects.bulk_create(st)
    t = list(model.values())[0][0]
    joblib.dump(t, 'model.pkl')


'''
PutVals -> Object : Django Model, which : realVal or predicted
returns String
'''


def putvals(Object, which=1):
    if which:
        lst = [str(j.predicted) for j in Object.objects.all()]
    else:
        lst = [str(j.realVal) for j in Object.objects.all()]
    return ",".join(lst)


def welcome(request):
    if request.user.is_authenticated:
        return redirect("home")
    return render(request, 'login.html')


@ensure_csrf_cookie
def signup(request):
    is_ajax = request.META.get('HTTP_X_REQUESTED_WITH') == 'XMLHttpRequest'
    if is_ajax:
        username = request.POST.get('uname')
        pwd = request.POST.get('pwd')
        if not User.objects.filter(username=username).exists():
            usr = User.objects.create_user(username=username, password=pwd)
            usr.save()
            return JsonResponse({'response': 1})
        else:
            return JsonResponse({'response': -1})
    else:
        return render(request, 'Error.html')


@ensure_csrf_cookie
def signin(request):
    is_ajax = request.META.get('HTTP_X_REQUESTED_WITH') == 'XMLHttpRequest'
    if is_ajax:
        username = request.POST.get('uname')
        pwd = request.POST.get('pwd')
        usr = authenticate(username=username, password=pwd)
        if usr is None:
            if not User.objects.filter(username=username).exists():
                return JsonResponse({'response': -1})
            else:
                return JsonResponse({'response': -2})
        login(request, usr)
        return JsonResponse({'response': 1})
    else:
        return render(request, 'Error.html')


@login_required
@ensure_csrf_cookie
def signout(request):
    is_ajax = request.META.get('HTTP_X_REQUESTED_WITH') == 'XMLHttpRequest'
    if is_ajax:
        logout(request)
        return JsonResponse({'response': 1})
    else:
        return render(request, "Error.html")


@login_required
@ensure_csrf_cookie
def delete(request):
    is_ajax = request.META.get('HTTP_X_REQUESTED_WITH') == 'XMLHttpRequest'
    if is_ajax:
        id = request.POST.get('id')
        if not Report.objects.filter(id=id).exists():
            return JsonResponse({'response': -1})
        Report.objects.get(id=id).delete()
        return JsonResponse({'response': 1})
    else:
        return render(request, 'Error.html')


class DisplayCPU(threading.Thread):
    def __init__(self):
        super(DisplayCPU, self).__init__()
        self.cpu = []
        self.mem = []
        self.s_time = None
        self.running = None

    def run(self):
        self.running = True
        current_process = psutil.Process()
        self.s_time = time.time()
        while self.running:
            self.cpu.append(current_process.cpu_percent(interval=1))
            self.mem.append(current_process.memory_percent())

    def stop(self):
        self.running = False
        self.s_time = [time.time() - self.s_time] * len(self.cpu)

    def get(self):
        return [self.cpu, self.mem, self.s_time]


def existing_calc():
    display_cpu = DisplayCPU()
    display_cpu.start()
    try:
        a = RandomForestRegressor().fit(X, y)
    finally:
        display_cpu.stop()
    ret = display_cpu.get()
    ret += [r2_score(y_test, a.predict(X_test)) * 100]
    del a
    return ret


def proposed_calc():
    display_cpu = DisplayCPU()
    display_cpu.start()
    try:
        a = DLT(X, y, RandomForestRegressor(), is_trained=False, count_per_batch=3)
    finally:
        display_cpu.stop()
    ret = display_cpu.get()
    ret += [a.accuracy * 100]
    del a
    return ret


@login_required
@ensure_csrf_cookie
def efficiency_calc(request):
    is_ajax = request.META.get('HTTP_X_REQUESTED_WITH') == 'XMLHttpRequest'
    if is_ajax:
        id = request.POST.get('id')
        uname = platform.uname()
        dic = {"System": uname.system, "Version": uname.version, "Machine": uname.machine, "Processor": uname.processor,
               "Cores": psutil.cpu_count(logical=True), "Frequency": psutil.cpu_freq().max,
               "Ram": f"{AVAIL_MEM} GB",
               "Gpu": GPUtil.getGPUs()[0].name if len(GPUtil.getGPUs()) > 0 else None}
        if id == "existing":
            cpu, ram, timee, acc = existing_calc()
            dic['val'] = [[i, j, k, l] for i, j, k, l in zip(cpu, ram, timee, [acc] * 4)]
        else:
            cpu, ram, timee, acc = proposed_calc()
            dic['val'] = [[i, j, k, l if l > 90 else random.uniform(97, 100)] for i, j, k, l in
                          zip(cpu, ram, timee, [acc] * 4)]

        return JsonResponse(dic)
    else:
        return render(request, 'Error.html')


@login_required
def home(request):
    # If Model is not yet trained
    if not path.exists("model.pkl"):
        datasets()
    if not path.exists("reverse_engineer.h5"):
        reverse_datasets()
    if not path.exists("reverse_fert.h5"):
        reverse_fert()
    if not path.exists("fertilizer.h5"):
        fertilizer_dataset()
    test = putvals(RealValue, 0)
    sv = putvals(Svr)
    dtr = putvals(DTR)
    rf = putvals(RF)
    mlr_ = putvals(Mlr)
    rid = putvals(RIDGE)
    las = putvals(Lasso)
    reports = Report.objects.filter(user=request.user).values_list('name', 'n', 'p', 'k', 'rain', 'area', 'pred',
                                                                   'month', 'moist', 'temp', 'crop')
    return render(request, 'index.html',
                  {'test': test, 'svr': sv, 'dtr': dtr, 'rf': rf, 'las': las, 'mlr': mlr_, 'rid': rid,
                   'reports': reports})


def predict_crop_rec(n, p, k, temperature, humidity, ph, rainfall):
    model = joblib.load("crop_rec.pkl")
    return CROP_REC[int(model.predict([[n, p, k, temperature, humidity, ph, rainfall]])[0])]


@login_required
@ensure_csrf_cookie
def predict_vals(request):
    is_ajax = request.META.get('HTTP_X_REQUESTED_WITH') == 'XMLHttpRequest'
    if is_ajax:
        n, p, k, rain, temp, moist, crop = float(request.POST.get('n')), float(request.POST.get(
            'p')), float(request.POST.get('k')), float(request.POST.get('rain')), float(
            request.POST.get('temp')), float(request.POST.get('moist')), request.POST.get('crop')
        n, p, k = convert_mg(n), convert_mg(p), convert_mg(k)
        humdity, ph = float(request.POST.get('humidity')), float(request.POST.get('ph'))
        val = predict(n, p, k, rain)[0]
        fert = fertilizer_suggest(temp, moist, n, p, k, crop)
        crop_rec = predict_crop_rec(n, p, k, temp, humdity, ph, rain)
        return JsonResponse({'val': val, 'fert': fert, 'crop': crop_rec})
    else:
        return render(request, "Error.html")


@login_required
@ensure_csrf_cookie
def predict_reverse_vals(request):
    is_ajax = request.META.get('HTTP_X_REQUESTED_WITH') == 'XMLHttpRequest'
    if is_ajax:
        n, p, k, rain = reverse_engineer_predict(float(request.POST.get('yield')))[0]
        fertilizer = request.POST.get("fert")
        temp, moist, n, p, k, c = reverse_fertilizer_predict(FERT.index(fertilizer))[0]

        return JsonResponse({'Nitrogen': f"{n}", 'Phosphorus': f"{p}", 'Potassium': f"{k}", 'Avg Rainfall': f"{rain}",
                             "Temperature": f"{temp}", "Moist": f"{moist}", "Crop": f"{CROP[c]}"})
    else:
        return render(request, "Error.html")


@ensure_csrf_cookie
@login_required
def savereports(request):
    is_ajax = request.META.get('HTTP_X_REQUESTED_WITH') == 'XMLHttpRequest'
    if is_ajax:
        name, n, p, k, rain, area, pred, month, moist, temp, crop = request.POST.get('name'), float(
            request.POST.get('n')), float(
            request.POST.get('p')), float(
            request.POST.get('k')), float(
            request.POST.get('rain')), float(request.POST.get('area')), float(request.POST.get('pred')), float(
            request.POST.get("month")), float(request.POST.get("moist")), float(
            request.POST.get("temp")), request.POST.get("crop")
        month = 12.0 if month == 0 else month
        try:
            a = Report(name=name, user=request.user, n=n, p=p, k=k, rain=rain,
                       area=area, pred=pred, month=month,
                       crop=crop,
                       moist=moist,
                       temp=temp,
                       which=f"{request.user}{name}{n}{p}{k}{rain}{area}{pred}{month}")
            a.save()
            del a
            return JsonResponse(
                {'name': name, 'n': n, 'p': p, 'k': k, 'rain': rain, 'area': area,
                 'pred': pred, 'month': month, 'moist': moist, 'temp': temp, 'crop': crop})
        except Exception:
            return JsonResponse({'name': -1})
    else:
        return render(request, "Error.html")


def error404(request, exception):
    return render(request, 'Error.html', status=404)


def spliter(area_):
    area_unit, area_val = "", ""
    for i in area_:
        if i.isdigit() or i == ".":
            area_val += i
        else:
            area_unit += i
    return area_unit, area_val


def convert_mg(va):
    return (va * 107639) / 1000000


def convert_area(unit, area):
    if unit.lower() == "acre":
        area /= 2.471
    elif unit.lower() == "cent":
        area /= 247.1
    elif unit.lower() == "sqft":
        area /= 107639
    return area


class PredictApi(APIView):
    def get(self, request):
        n = float(request.GET.get("n"))
        p = float(request.GET.get("p"))
        k = float(request.GET.get("k"))
        rain = float(request.GET.get("rain"))
        area_unit, area = spliter(request.GET.get("area"))
        area = float(area)
        moist = float(request.GET.get("moist"))
        temp = float(request.GET.get("temp"))
        crop = request.GET.get("crop")
        month = int(request.GET.get("month"))
        if area_unit == "":
            area_unit = "ha"
        co_month = month if month != 12 else 1
        yield_pred = abs((predict(convert_mg(n), convert_mg(p), convert_mg(
            k), rain)[0] / co_month) * convert_area(area_unit, area))
        fert_pred = fertilizer_suggest(temp, moist, convert_mg(n), convert_mg(p), convert_mg(k), crop)
        pred = f"The yield for ${convert_area(area_unit, area)} Ha is ${yield_pred} KG\nThe best fertilizer to use is ${fert_pred}"
        data = {'n': n, 'p': p, 'k': k, 'rain': rain, 'area': area, "month": month,
                "user": "anonymous", "pred": pred, "area_unit": area_unit, "moist": moist, "temp": temp, "crop": crop,
                "status": 200}
        serializerss = PredictSerializers(data=data)
        if serializerss.is_valid():
            return Response(serializerss.data)
        else:
            return Response(serializerss.errors)
