from django.test import TestCase
from django.urls import reverse
from django.contrib.auth.models import User
import ast
# Create your tests here.
class TestViews(TestCase):

    def setUp(self):
        self.user = User.objects.create_user(username='ak@gmail.com', password='123')
        self.client.login(username='ak@gmail.com', password='123')


    def test_starting_page(self):
        res = self.client.get(reverse("welcome"),follow=True)
        self.assertEqual(res.status_code,200)
    
    def test_register_page(self):
        res = self.client.post(reverse("signup"),{'uname':"manthirajak@gmail.com","pwd":"123"}, **{'HTTP_X_REQUESTED_WITH': 'XMLHttpRequest'})
        self.assertEqual(res.status_code, 200)
        response = res.content.decode("UTF-8")
        code = ast.literal_eval(response)
        self.assertEqual(code["response"],1)

    def test_login_page(self):
        res = self.client.post(reverse("signin"),{'uname':"manthirajak@gmail.com","pwd":"123"}, **{'HTTP_X_REQUESTED_WITH': 'XMLHttpRequest'})
        self.assertEqual(res.status_code,200)
        response = res.content.decode("UTF-8")
        code = ast.literal_eval(response)
        self.assertEqual(code["response"],-1)

    def test_predict_page(self):
        res = self.client.post(reverse("predict"),{'n':23.1,'p':24.0,'k':26.4,'rain':152},**{'HTTP_X_REQUESTED_WITH': 'XMLHttpRequest'})
        self.assertEqual(res.status_code, 200)
        print(res.content)
    
    def test_save_report_page(self):
        res = self.client.post(reverse("save"),{'n':23.1,'p':24.0,'k':26.4,'rain':152,'month':3,'area':5,'pred':1751.124},**{'HTTP_X_REQUESTED_WITH': 'XMLHttpRequest'})
        self.assertEqual(res.status_code,200)
        print(res.content)

    def test_delete_report_page(self):
        res = self.client.post(reverse("delete"),{'id':1},**{'HTTP_X_REQUESTED_WITH': 'XMLHttpRequest'})
        self.assertEqual(res.status_code, 200)
        print(res.content)

    def test_signout(self):
        res = self.client.post(reverse("logout"),{},**{'HTTP_X_REQUESTED_WITH': 'XMLHttpRequest'})
        self.assertEqual(res.status_code, 200)
        response = res.content.decode("UTF-8")
        code = ast.literal_eval(response).get("response")
        self.assertEqual(code,1)