from operator import mod
from django.db import models
from django.db.models.base import Model


class Mlmodel(models.Model):
    class Meta:
        verbose_name_plural = "MlModel"
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


class mlr(models.Model):
    class Meta:
        verbose_name_plural = "mlr"
    predicted = models.FloatField()


class lasso(models.Model):
    class Meta:
        verbose_name_plural = "lasso"
    predicted = models.FloatField()


class ridge(models.Model):
    class Meta:
        verbose_name_plural = "ridge"
    predicted = models.FloatField()


class RealValue(models.Model):
    class Meta:
        verbose_name_plural = "RealValue"
    realVal = models.FloatField()


class Report(models.Model):
    class Meta:
        verbose_name_plural = "Report"
    user = models.TextField()
    which = models.TextField(unique=True)
    n = models.FloatField()
    p = models.FloatField()
    k = models.FloatField()
    rain = models.FloatField()
    area = models.FloatField()
    pred = models.FloatField()
