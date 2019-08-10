#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2019/7/23
# @Author  : Edwin
# @Version : Python 3.6
# @File    : fetch_mysql_knob_data.py


import pymysql
from fetch_knob_metric import mysql_database as database
from fetch_knob_metric.JSONUtile import JSONUtil
import time
import os
import subprocess


class fetch_mysql_knob():

    def __init__(self):
        self.__conn = pymysql.connect(host=database.WL_HOST,port=database.WL_PORT,user=database.WL_USER,password=database.WL_PASSWD,db=database.WL_DB)
        self.__cur = self.__conn.cursor()
        self.__cur.execute("SET NAMES UTF8")


    def get_mysql_knob(self):
        command = "SHOW VARIABLES;"
        self.__cur.execute(command)
        result = self.__cur.fetchall()

        res_ = {"global":dict(result)}
        res_ = {"global":dict(res_),"local":None}

        self.__conn.close()
        os.chdir("/oracle_predict/")

        if not os.path.isdir(time.strftime("%d_%m_%Y")):
            os.mkdir(time.strftime("%d_%m_%Y"))

        os.chdir("/oracle_predict/"+time.strftime("%d_%m_%Y"))
        #filename = "knob_connfig_" + time.strftime("%d_%m_%Y") + ".json"
        filename = "knobs.json"
        with open(filename, 'w') as file_obj:
            JSONUtil.dump(res_, file_obj)