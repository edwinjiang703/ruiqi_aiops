#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2019/8/1
# @Author  : Edwin
# @Version : Python 3.6
# @File    : upload_oracle_metriccatalog.py


import pymysql
from fetch_knob_metric import mysql_database as database
from fetch_knob_metric.JSONUtile import JSONUtil
import  time,os
import logging,json

class upload_ora_metric():

    def __init__(self):
        try:
            self.__conn = pymysql.connect(host=database.OT_HOST, port=database.OT_PORT, user=database.OT_USER,
                                          password=database.OT_PASSWD, db=database.OT_DB)
            self.__cur = self.__conn.cursor()
            self.__cur.execute("SET NAMES UTF8")

        except Exception as e:
            logging.info(e)

    def upload_metric_metadata(self):

        os.chdir("/oracle_predict/oracle_data/" + time.strftime("%d_%m_%Y"))
        with open ('metric_before.json','r') as file_:
            result = json.load(file_)

        self.__cur.execute("USE ottertune")

        for scope, sub_vars in list(result.items()):
            if sub_vars is None:
                continue
            if scope == 'global':
                for view_name, variables in list(sub_vars.items()):
                    for var_name, var_value in list(variables.items()):
                        sql_ = """insert into website_metriccatalog (`name`,vartype,summary,scope,metric_type,dbms_id)
                                  values(%s,%s,%s,%s,%s,%s)
                               """
                        self.__cur.execute(sql_,
                                           [var_name,1,'','global',1,13])
        self.__conn.commit()