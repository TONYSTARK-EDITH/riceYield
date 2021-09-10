from . import views
from django.urls import path

urlpatterns = [
    path('', views.welcome,name="welcome"),
    path('home',views.home,name="home"),
    path('predict',views.Predict,name="predict"),
    path('save',views.saveReports,name="save"),
    path('register',views.signup,name="signup"),
    path('signin',views.signin,name="signin"),
    path('logout',views.signout,name="logout"),
    path('delete',views.delete,name="delete"),
    path('report',views.exampleAPI.as_view(),name="example"),
]
