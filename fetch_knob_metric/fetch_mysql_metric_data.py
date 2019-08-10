#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2019/7/23
# @Author  : Edwin
# @Version : Python 3.6
# @File    : fetch_mysql_metric_data.py

import pymysql
from fetch_knob_metric import mysql_database as database
from fetch_knob_metric.JSONUtile import JSONUtil
import  time,os
import json
class fetch_mysql_metric():

    def __init__(self):
        self.__conn = pymysql.connect(host=database.WL_HOST,port=database.WL_PORT,user=database.WL_USER,password=database.WL_PASSWD,db=database.WL_DB)
        self.__cur = self.__conn.cursor()
        self.__cur.execute("SET NAMES UTF8")


    def get_mysql_metric(self,flag,period):
        command = "SHOW STATUS;"
        self.__cur.execute(command)
        result = self.__cur.fetchall()

        res_ = {"global":dict(result)}
        res_ = {"global":dict(res_),"local":None}

        self.__conn.close()
        os.chdir("/oracle_predict/")

        if not os.path.isdir(time.strftime("%d_%m_%Y")):
            os.mkdir(time.strftime("%d_%m_%Y"))

        os.chdir("/oracle_predict/"+time.strftime("%d_%m_%Y"))

        #filename = "metrics_"+flag+"-"+period+".metrics"
        filename = "metrics_" + flag + ".json"
        with open(filename,'w') as file_obj:
           JSONUtil.dump(res_,file_obj)
