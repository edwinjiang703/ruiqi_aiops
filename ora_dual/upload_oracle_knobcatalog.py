#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2019/7/31
# @Author  : Edwin
# @Version : Python 3.6
# @File    : upload_oracle_knobcatalog.py


import pymysql
from fetch_knob_metric import mysql_database as database
from fetch_knob_metric.JSONUtile import JSONUtil
import  time,os
import logging

class upload_ora_knob():

    def __init__(self):
        try:
            self.__conn = pymysql.connect(host=database.OT_HOST, port=database.OT_PORT, user=database.OT_USER,
                                          password=database.OT_PASSWD, db=database.OT_DB)
            self.__cur = self.__conn.cursor()
            self.__cur.execute("SET NAMES UTF8")
        except Exception as e:
            logging.info(e)

    def upload_parm_metadata(self):

        try:
            os.chdir("/oracle_predict/" + time.strftime("%d_%m_%Y"))
            result = JSONUtil.loads('oracle_parameter.json')
            print(result)
        except Exception as e:
            logging.info(e)

