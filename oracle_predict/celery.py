# -*- coding: utf-8 -*-
# @Time    : 2019/1/10 11:21 PM
# @Author  : Edwin
# @File    : celery.py
# @Software: PyCharm


import os
from celery import Celery

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "oracle_predict.settings")

from django.conf import settings

app = Celery('oracle_predict')
app.config_from_object('django.conf:settings')
app.autodiscover_tasks(lambda: settings.INSTALLED_APPS)

@app.task(bind=True)
def debug_task(self):
    print('Request: {0!r}'.format(self.request))



