from django.db import models

# # Create your models here.
class Mlmodel(models.Model):
    class Meta:
        verbose_name_plural="MlModel"
    nitrogen = models.FloatField()
    phosphorus = models.FloatField()
    pottasium = models.FloatField()
    rainfall = models.FloatField()
    rice_yield = models.FloatField()

class svr(models.Model):
    class Meta:
        verbose_name_plural = "SVR"
    predicted = models.FloatField()

class DTR(models.Model):
    class Meta:
        verbose_name_plural = "DTR"
    predicted = models.FloatField()
class RF(models.Model):
    class Meta:
        verbose_name_plural = "RF"
    predicted = models.FloatField()

class RealValue(models.Model):
    class Meta:
        verbose_name_plural = "RealValue"
    realVal = models.FloatField()