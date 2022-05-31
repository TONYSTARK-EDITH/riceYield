from django.test import TestCase
from classifier.models import *

class TestModels(TestCase):
    def test_Mlmodel(self):
        Mlmodel.objects.bulk_create([Mlmodel(nitrogen=1,pottasium=1,phosphorus=1,rainfall=1,rice_yield=5)])
        length = len(list(Mlmodel.objects.all()))
        self.assertEqual(length,1)
        print(f"MlModle ------- {length}")
    
    def test_svr(self):
        Svr.objects.bulk_create([Svr(predicted=215)])
        length = len(list(Svr.objects.all()))
        self.assertEqual(length,1)
        print(f"SVR ------- {length}")
    
    def test_mlr(self):
        Mlr.objects.bulk_create([Mlr(predicted=215)])
        length = len(list(Mlr.objects.all()))
        self.assertEqual(length,1)
        print(f"MLR ------- {length}")

    def test_DTR(self):
        DTR.objects.bulk_create([DTR(predicted=215)])
        length = len(list(DTR.objects.all()))
        self.assertEqual(length,1)
        print(f"DTR ------- {length}")  

    def test_RF(self):
        RF.objects.bulk_create([RF(predicted=215)])
        length = len(list(RF.objects.all()))
        self.assertEqual(length,1)
        print(f"RF ------- {length}")
    
    def test_lasso(self):
        Lasso.objects.bulk_create([Lasso(predicted=215)])
        length = len(list(Lasso.objects.all()))
        self.assertEqual(length,1)
        print(f"LASSO ------- {length}")
    
    def test_ridge(self):
        RIDGE.objects.bulk_create([RIDGE(predicted=215)])
        length = len(list(RIDGE.objects.all()))
        self.assertEqual(length,1)
        print(f"RIDGE ------- {length}")      
    
    def test_RealValue(self):
        RealValue.objects.bulk_create([RealValue(realVal=215)])
        length = len(list(RealValue.objects.all()))
        self.assertEqual(length,1)
        print(f"REAL VALUE ------- {length}")
    
    def test_Report(self):
        Report.objects.bulk_create([Report(user="12",which="1",n=1,p=2,k=3,rain=4,area=6,pred=2,month=3)])
        length = len(list(Report.objects.all()))
        self.assertEqual(length,1)
        print(f"REPORT ------- {length}")