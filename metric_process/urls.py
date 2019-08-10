#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2019/1/14
# @Author  : Edwin
# @Version : Python 3.6
# @File    : urls.py

from  metric_process import views
from django.conf.urls import url,re_path,include
from django.contrib import admin
from metric_process import space_predict_122
urlpatterns = [
    #FBV Function base view
    url(r'^admin/', admin.site.urls),
    url(r'^process_spacechange_data/',space_predict_122.process_space_metric_data),
    url(r'^spacechange_trend/',space_predict_122.analyze_space_change),
    url(r'^spacechange_arima/',space_predict_122.arima_spacechange_trend),
    url(r'^fa_analyze/',space_predict_122.Fa_kmeans),
    url(r'^sapreport/',space_predict_122.sapreport)
    ]