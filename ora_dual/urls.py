#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2019/1/14
# @Author  : Edwin
# @Version : Python 3.6
# @File    : urls.py

from django.conf.urls import url,include
from django.contrib import admin
from ora_dual import views
from ora_dual import display_charts,iops_thu_data
from fetch_knob_metric.fetch_mysql_knob_data import fetch_mysql_knob
from fetch_knob_metric.fetch_mysql_metric_data import fetch_mysql_metric


urlpatterns = [
    #FBV Function base view
    url(r'^admin/', admin.site.urls),
    url(r'^index/',views.index),
    url(r'^task_res/',views.task_res),
    url(r'^welcome/',views.login),
    url(r'^login/',views.login),
    #url(r'^home/',views.home,name='ihome'),#name 是别名，可以在view里面替换。
    url(r'^home/', views.home),
    url(r'^sapreport/', views.sapreport),
    #CBV Class Base View
    url(r'^TestCBV/',views.TestCBV.as_view()),
    url(r'^ormtest/',views.ormtest),
    #re_path('detail/(<nid>\d+)/',views.detail)
    url(r'^detail/',views.detail),
    url(r'^delete/',views.delete),
    url(r'^get_load_profile_data/',views.get_load_profile_data),
    url(r'^sel_metric_data/',views.sel_metric_data),
    url(r'^load_profile_trend/',display_charts.bar_echart),
    url(r'^get_status_data/',iops_thu_data.iops_echart),
    url(r'^get_mysql_knob/', views.get_mysql_knob),
    url(r'^get_mysql_metric/', views.get_mysql_metric),

    url(r'^upload_data/', views.upload_data),

    url(r'^get_oraparam_metadata/', views.get_ora_parameter_metadata),
    url(r'^upload_oraparam_metadata/', views.upload_ora_parameter_metadata),

    url(r'^get_oracle_metric/', views.get_oracle_metric),
    url(r'^upload_orametric_metadata/', views.upload_oracle_metric_metadata),
    url(r'^detect_value',views.detect_metis_value)



    #url(r'ora_dual/', include('ora_dual.urls'))
    #引入正则表达式
    #re_path('detail-(?P<uid>\d+)-(?P<id>\d+).html/',views.detail)
    #url(r'^metric_process/',include('metric_process.url'))
]