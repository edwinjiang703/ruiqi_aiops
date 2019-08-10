#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2019/7/23
# @Author  : Edwin
# @Version : Python 3.6
# @File    : fetch_mysql.py
from fetch_knob_metric.fetch_mysql_knob_data import fetch_mysql_knob
from fetch_knob_metric.fetch_mysql_metric_data import fetch_mysql_metric
from django.http import HttpResponse
from django.shortcuts import render
from django.template import loader

def get_mysql_knob(request):
    knob_ = fetch_mysql_knob()
    knob_.get_mysql_knob()
    #return render(request, "./node_modules/gentelella/production/home.html")
    template = loader.get_template('pyecharts.html')
    context = []
    return HttpResponse(template.render(context, request))

def get_mysql_metric(request):
    metric_ = fetch_mysql_metric()
    metric_.get_mysql_metric()
    return render(request, "./node_modules/gentelella/production/home.html")
