#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2019/8/1
# @Author  : Edwin
# @Version : Python 3.6
# @File    : fetch_oracle_metric_data.py


import cx_Oracle
import logging
from fetch_knob_metric import oracle_setting
import os,time
from fetch_knob_metric.JSONUtile import JSONUtil

class fetch_oracle_metic_data():
    def __init__(self):
        self._conn ,self._cur = self.get_conn_and_cursor()

    def get_conn_and_cursor(self):
        try:
            conn = cx_Oracle.connect(oracle_setting.DATA_SOURCE_NAME)
            curs = conn.cursor()
            return conn, curs
        except Exception as e:
            logging.error(str(e))

    def close_db(self):
        # 关闭游标
        self._conn.close()
        # 关闭DB连接
        self._cur.close()


    def collect_metic_data(self,flag):

        os.chdir("/oracle_predict/oracle_data/")

        sql="""
            select "Metric", 
                   case when "Total" >10000000 then ''||round("Total"/1024/1024,0)
                        when "Total" between 10000 and 10000000 then ''||round("Total"/1024,0)
                        when "Total" between 10 and 1024 then to_char(round("Total",0))||''
                        else ''||to_char("Total") 
                   end "Total"
            from (
             select to_char(min(begin_time),'hh24:mi:ss')||' /'||round(avg(intsize_csec/100),0)||'s' "Time_Delta",
                   metric_name||' - '||metric_unit "Metric", 
                   nvl(sum(value_inst1),0)+nvl(sum(value_inst2),0) "Total",
                   sum(value_inst1) inst1, sum(value_inst2) inst2
             from
              ( select begin_time,intsize_csec,metric_name,metric_unit,metric_id,group_id,
                   case inst_id when 1 then round(value,1) end value_inst1,
                   case inst_id when 2 then round(value,1) end value_inst2
              from gv$sysmetric 
              where metric_name in (select metric_name from v$metricname where  group_id=2) and group_id=2)
             group by metric_id,group_id,metric_name,metric_unit
             order by metric_name)
        """
        if not os.path.isdir(time.strftime("%d_%m_%Y")):
            os.mkdir(time.strftime("%d_%m_%Y"))

        os.chdir("/oracle_predict/oracle_data/" + time.strftime("%d_%m_%Y"))

        self._cur.execute(sql)
        result = self._cur.fetchall()

        res_ = {"global": dict(result)}
        res_ = {"global": dict(res_), "local": None}

        filename = "metrics_"+flag+".json"
        with open(filename, 'w') as file_obj:
            JSONUtil.dump(res_, file_obj)