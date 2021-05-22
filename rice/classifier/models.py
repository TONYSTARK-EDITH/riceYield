from django.db import models

# Create your models here.
class Mlmodel(models.Model):
    class Meta:
        verbose_name_plural="MlModel"
    pottasium = models.FloatField()
    phosphorus = models.FloatField()
    nitrogen = models.FloatField()
    rainfall = models.FloatField()
    rice_yield = models.FloatField()

class SVR(models.Model):
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