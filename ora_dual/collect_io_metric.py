# -*- coding: utf-8 -*-
# @Time    : 2019/1/10 10:11 PM
# @Author  : Edwin
# @File    : collect_io_metric.py
# @Software: PyCharm

from celery.task import periodic_task
from celery.utils.log import get_task_logger
import json

# Log debug messages
LOG = get_task_logger(__name__)
@periodic_task(run_every=5, name="collect_io_task")
def collect_io_task():
    LOG.info("test")
    data = {'test':1}
    with open('test.json',w) as f:
        json.dump(data,f)



