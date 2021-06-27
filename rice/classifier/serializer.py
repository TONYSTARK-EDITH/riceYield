from rest_framework import serializers
from .models import Report

class predictSerializers(serializers.Serializer):
    user = serializers.CharField()
    n = serializers.FloatField()
    p = serializers.FloatField()
    k = serializers.FloatField()
    rain = serializers.FloatField()
    area = serializers.FloatField()
    pred = serializers.FloatField()
    month = serializers.FloatField()
    area_unit = serializers.CharField()
    status = serializers.IntegerField()

