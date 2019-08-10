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
import logging,json

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

        os.chdir("/oracle_predict/oracle_data/" + time.strftime("%d_%m_%Y"))
        with open ('knobs.json','r') as file_:
            result = json.load(file_)

        self.__cur.execute("USE ottertune")

        for scope, sub_vars in list(result.items()):
            if sub_vars is None:
                continue
            if scope == 'global':
                for view_name, variables in list(sub_vars.items()):
                    for var_name, var_value in list(variables.items()):
                        print(type(var_value))
                        if var_value is None:
                            var_value = ' '
                        if  (var_value.isdigit()):
                            sql_ = """insert into website_knobcatalog (`name`,vartype,unit,category,summary,description,scope,minval,maxval,`default`,enumvals,context,tunable,resource,dbms_id)
                                      values(%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
                                   """
                            self.__cur.execute(sql_,
                                               [var_name, 3, 3, '', '', '', 'global', 1, 65535, var_value, None, 'dynamic',
                                                1, 4, 13])
                            #(name,vartype,unit,category,summary,descroption,scope,minval,maxval,default,enumvals,context,tunable,resource,dbms_id)
                        else:
                            sql_ = """insert into website_knobcatalog(`name`,vartype,unit,category,summary,description,scope,minval,maxval,`default`,enumvals,context,tunable,resource,dbms_id) 
                                      values(%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
                                   """
                            self.__cur.execute(sql_,
                                               [var_name, 4, 3, '', '', '', 'global', 1, 65535, var_value, None, 'dynamic',
                                                1, 4, 13])
                        print(var_name, var_value)
        self.__conn.commit()

