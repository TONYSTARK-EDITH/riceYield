from . import views
from django.urls import path

urlpatterns = [
    path('', views.home),
    path('predict',views.Predict),
]
