from django.contrib import admin

from .models import *
admin.site.register(Mlmodel)
admin.site.register(svr)
admin.site.register(DTR)
admin.site.register(RF)
admin.site.register(RealValue)