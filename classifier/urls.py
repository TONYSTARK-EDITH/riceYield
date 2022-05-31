from . import views
from django.urls import path

urlpatterns = [
    path('', views.welcome, name="welcome"),
    path('home', views.home, name="home"),
    path('predict', views.predict_vals, name="predict"),
    path('save', views.savereports, name="save"),
    path('register', views.signup, name="signup"),
    path('signin', views.signin, name="signin"),
    path('logout', views.signout, name="logout"),
    path('delete', views.delete, name="delete"),
    path('rev_pred', views.predict_reverse_vals, name="reverse"),
    path('apipredict', views.PredictApi.as_view(), name="api"),
    path('test', views.efficiency_calc, name="test")
]
