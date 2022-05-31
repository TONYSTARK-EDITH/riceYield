from django.test import TestCase
from django.urls import reverse,resolve
from classifier.views import *

class TestUrls(TestCase):
    def test_welcome_page(self):
        url = reverse("welcome")
        resolved_url = resolve(url)
        print(resolved_url.func)
        self.assertEqual(resolved_url.func, welcome)
    
    def test_home_page(self):
        url = reverse("home")
        resolved_url = resolve(url)
        print(resolved_url.func)
        self.assertEqual(resolved_url.func, home)

    def test_signup_page(self):
        url = reverse("signup")
        resolved_url = resolve(url)
        print(resolved_url.func)
        self.assertEqual(resolved_url.func, signup)

    def test_signin_page(self):
        url = reverse("signin")
        resolved_url = resolve(url)
        print(resolved_url.func)
        self.assertEqual(resolved_url.func, signin)

    def test_predict_page(self):
        url = reverse("predict")
        resolved_url = resolve(url)
        print(resolved_url.func)
        self.assertEqual(resolved_url.func, predict_vals)

    def test_save_page(self):
        url = reverse("save")
        resolved_url = resolve(url)
        print(resolved_url.func)
        self.assertEqual(resolved_url.func, savereports)

    def test_delete_page(self):
        url = reverse("delete")
        resolved_url = resolve(url)
        print(resolved_url.func)
        self.assertEqual(resolved_url.func, delete)

    def test_logout_page(self):
        url = reverse("logout")
        resolved_url = resolve(url)
        print(resolved_url.func)
        self.assertEqual(resolved_url.func, signout)