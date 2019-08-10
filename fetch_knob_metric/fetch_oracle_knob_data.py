#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2019/7/30
# @Author  : Edwin
# @Version : Python 3.6
# @File    : fetch_oracle_knob_data.py
import cx_Oracle
import logging
from fetch_knob_metric import oracle_setting
import os,time
from fetch_knob_metric.JSONUtile import JSONUtil

class fetch_oracle_knob_data():
    def __init__(self):
        self._conn ,self._cur = self.get_conn_and_cursor()

    def get_conn_and_cursor(self):

        conn = cx_Oracle.connect(oracle_setting.DATA_SOURCE_NAME)
        curs = conn.cursor()
        return conn, curs


    def close_db(self):
        # 关闭游标
        self._conn.close()
        # 关闭DB连接
        self._cur.close()


    def collect_parameter_data(self):
        # Parameter
        # 1 - Boolean
        # 2 - String
        # 3 - Integer
        # 4 - Parameter file
        # 5 - Reserved
        # 6 - Big integer

        os.chdir("/oracle_predict/oracle_data/")

        sql="""
            select name,nvl(value,0) from v$parameter where type in (1,3,6)
        """
        if not os.path.isdir(time.strftime("%d_%m_%Y")):
            os.mkdir(time.strftime("%d_%m_%Y"))

        os.chdir("/oracle_predict/oracle_data/" + time.strftime("%d_%m_%Y"))

        self._cur.execute(sql)
        result = self._cur.fetchall()
        print(result)

        res_ = {"global": dict(result)}
        res_ = {"global": dict(res_), "local": None}

        filename = "knobs.json"
        with open(filename, 'w') as file_obj:
            JSONUtil.dump(res_, file_obj)
