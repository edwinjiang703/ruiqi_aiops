#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2019/1/23
# @Author  : Edwin
# @Version : Python 3.6
# @File    : collect_metric.py

from ora_dual import models
from celery.task import periodic_task
import subprocess

@periodic_task(run_every=5, name="run_cmd_out")
#@shared_task
def run_cmd_out(cmd="df -h"):
    cmd_out = subprocess.Popen(cmd,shell=True,stdout=subprocess.PIPE,stderr=subprocess.PIPE)
    return cmd_out.stdout.read()
    #print(result.get())