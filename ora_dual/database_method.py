#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2019/1/17
# @Author  : Edwin
# @Version : Python 3.6
# @File    : database_method.py
import cx_Oracle
class initial_connect():
    def __init__(self,username,pwd,conn_str):
        self.username = username
        self.pwd = pwd
        self.conn_str = conn_str

    def create_conn(self):
        self.conn = cx_Oracle.connect(self.username+'/'+self.pwd+'@'+self.conn_str)
        return self.conn