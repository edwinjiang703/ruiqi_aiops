# -*- coding: utf-8 -*-
# @Time    : 2019/1/21 10:33 PM
# @Author  : Edwin
# @File    : iops_thu_data.py
# @Software: PyCharm

from django.http import HttpResponse
from django.template import loader
from pyecharts import Line3D,Bar,Timeline,Pie
from ora_dual import models
from celery.task import periodic_task
from ora_dual import database_method
import numpy as np
import pandas as pd
import cx_Oracle

REMOTE_HOST = '/static/assets/js'

#@periodic_task(run_every=5, name="iops_echart")
def iops_echart(request):

    template = loader.get_template('display_iops_thu.html')
    #conn = cx_Oracle.connect('system/oracle@trn')
    try:
        conn = database_method.initial_connect('system', 'oracle', 'trn')
        conn = conn.create_conn()
    except Exception as msg:
        print(msg)


    try:


        iops_tho_sec = """
                select "Time_Delta", "Metric", 
                   case when "Total" >10000000 then '* '||round("Total"/1024/1024,0)||' M' 
                        when "Total" between 10000 and 10000000 then '+ '||round("Total"/1024,0)||' K'
                        when "Total" between 10 and 1024 then '  '||to_char(round("Total",0))
                        else '  '||to_char("Total") 
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
              where metric_name in ('Host CPU Utilization (%)','Current OS Load', 'Physical Write Total IO Requests Per Sec', 
                    'Physical Write Total Bytes Per Sec', 'Physical Write IO Requests Per Sec', 'Physical Write Bytes Per Sec',
                     'I/O Requests per Second', 'I/O Megabytes per Second',
                    'Physical Read Total Bytes Per Sec', 'Physical Read Total IO Requests Per Sec', 'Physical Read IO Requests Per Sec',
                    'CPU Usage Per Sec','Network Traffic Volume Per Sec','Logons Per Sec','Redo Generated Per Sec','Redo Writes Per Sec',
                    'User Transaction Per Sec','Average Active Sessions','Average Synchronous Single-Block Read Latency',
                    'Logical Reads Per Sec','DB Block Changes Per Sec') and group_id=2
              )
             group by metric_id,group_id,metric_name,metric_unit
             order by metric_name
            )
                """

        instance_status ="""
        select instance_name,status,to_char(startup_time,'yyyy-mm-dd:HH24:MI:SS') startup_time from gv$instance
        """

        cursor = conn.cursor()
        cursor.execute(iops_tho_sec)
        iops_tho_sec_data = cursor.fetchall()

        iops_title = [i[0] for i in cursor.description]
        #iops_tho_data = pd.DataFrame(np.array(iops_tho_sec_data), columns=iops_title)
        iops_tho_data = []

        for idx in range(len(iops_tho_sec_data)):
            iops_tho_data.append({iops_title[0]:iops_tho_sec_data[idx][0],iops_title[1]:iops_tho_sec_data[idx][1],iops_title[2]:iops_tho_sec_data[idx][2]})
        #print(iops_tho_data)
        cursor.close()

        cursor = conn.cursor()
        cursor.execute(instance_status)
        instance_status_data = cursor.fetchall()
        inst_status_data = []

        inst_title = [i[0] for i in cursor.description]

        total = 0
        good = 0
        for idx in range(len(instance_status_data)):
            inst_status_data.append({inst_title[0]:instance_status_data[idx][0],inst_title[1]:instance_status_data[idx][1],inst_title[2]:instance_status_data[idx][2]})
            if instance_status_data[idx][1] == 'OPEN':
                good +=1
            total += 1

        health = good/total
        #print(health)


        # #水球图
        from pyecharts import Liquid

        liquid_instance = Liquid(width=200,height=50)
        liquid_instance.add("Liquid",[health])

        context = dict(
            iops_title =iops_title,
            iops_tho_data = iops_tho_data,
            inst_title = inst_title,
            instance_status_data = inst_status_data,
            msg = '',
            health_ball=liquid_instance.render_embed(),
            host=REMOTE_HOST,
            script_list = liquid_instance.get_js_dependencies()
        )
        return HttpResponse(template.render(context, request))
    except Exception as msg:
        context = dict(
            msg = msg,
            sid = 'TRN Instance'
        )
        return HttpResponse(template.render(context, request))