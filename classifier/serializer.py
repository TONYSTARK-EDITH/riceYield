from rest_framework import serializers


class PredictSerializers(serializers.Serializer):
    user = serializers.CharField()
    n = serializers.FloatField()
    p = serializers.FloatField()
    k = serializers.FloatField()
    rain = serializers.FloatField()
    area = serializers.FloatField()
    pred = serializers.CharField()
    month = serializers.FloatField()
    moist = serializers.FloatField()
    temp = serializers.FloatField()
    crop = serializers.CharField()
    area_unit = serializers.CharField()
    status = serializers.IntegerField()
