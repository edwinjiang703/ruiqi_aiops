#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2019/7/23
# @Author  : Edwin
# @Version : Python 3.6
# @File    : main.py

from fetch_knob_metric.fetch_mysql_knob_data import fetch_mysql_knob
from fetch_knob_metric.fetch_mysql_metric_data import fetch_mysql_metric
knob_ = fetch_mysql_knob()
metric_=fetch_mysql_metric()

if __name__ == '__main__':
    knob_.get_mysql_knob()
    metric_.get_mysql_metric()