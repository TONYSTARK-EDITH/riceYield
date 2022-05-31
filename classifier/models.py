from django.db import models


class Mlmodel(models.Model):
    class Meta:
        verbose_name_plural = "MlModel"

    nitrogen = models.FloatField()
    phosphorus = models.FloatField()
    pottasium = models.FloatField()
    rainfall = models.FloatField()
    rice_yield = models.FloatField()


class Svr(models.Model):
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


class Mlr(models.Model):
    class Meta:
        verbose_name_plural = "mlr"

    predicted = models.FloatField()


class Lasso(models.Model):
    class Meta:
        verbose_name_plural = "lasso"

    predicted = models.FloatField()


class RIDGE(models.Model):
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

    name = models.TextField()
    which = models.TextField(unique=True)
    user = models.TextField()
    n = models.FloatField()
    p = models.FloatField()
    k = models.FloatField()
    rain = models.FloatField()
    area = models.FloatField()
    pred = models.FloatField()
    month = models.FloatField()
    moist = models.FloatField()
    temp = models.FloatField()
    crop = models.TextField()
