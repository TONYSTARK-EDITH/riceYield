from . import views
from django.urls import path

urlpatterns = [
    path('', views.welcome),
    path('home',views.home,name="home"),
    path('predict',views.Predict),
    path('save',views.saveReports),
    path('register',views.signup),
    path('signin',views.signin),
    path('logout',views.signout),
    path('delete',views.delete),
]
