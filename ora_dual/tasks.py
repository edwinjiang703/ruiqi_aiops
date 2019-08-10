#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2019/1/11
# @Author  : Edwin
# @Version : Python 3.6
# @File    : tasks.py

from __future__ import absolute_import, unicode_literals
from celery import shared_task
from celery.task import periodic_task
from ora_dual import database_method
from ora_dual import models
import pandas as pd
import numpy as np

from fetch_knob_metric.fetch_mysql_metric_data import fetch_mysql_metric
from fetch_knob_metric.JSONUtile import JSONUtil
import os,time,datetime,random

@shared_task
def add(x, y):
    return x + y


@shared_task
def mul(x, y):
    return x * y

@shared_task
def test():
    return "hello world"


@shared_task
def xsum(numbers):
    return sum(numbers)

# @periodic_task(run_every=5, name="run_cmd")
# #@shared_task
# def run_cmd(cmd="df -h"):
#     cmd_out = subprocess.Popen(cmd,shell=True,stdout=subprocess.PIPE,stderr=subprocess.PIPE)
#     return cmd_out.stdout.read()
#     #print(result.get())

@periodic_task(run_every = 3600,name = "collect_diskspace")
def collect_diskspace():

    conn = database_method.initial_connect('system', 'oracle', 'trn')
    conn = conn.create_conn()

    space_usage = """
        SELECT 
        to_char(sysdate,'yyyy-mm-dd') data_time,
        A.TABLESPACE_NAME tablespace_name,
        A.TOTAL_SPACE total,
        NVL(B.FREE_SPACE, 0) free,
        A.TOTAL_SPACE - NVL(B.FREE_SPACE, 0) used,
        CASE WHEN A.TOTAL_SPACE=0 THEN 0 ELSE trunc(NVL(B.FREE_SPACE, 0) / A.TOTAL_SPACE * 100, 2) END percent
    FROM (SELECT TABLESPACE_NAME, trunc(SUM(BYTES) / 1024 / 1024/1024 ,2) TOTAL_SPACE
          FROM DBA_DATA_FILES
         GROUP BY TABLESPACE_NAME) A,
       (SELECT TABLESPACE_NAME, trunc(SUM(BYTES / 1024 / 1024/1024  ),2) FREE_SPACE
          FROM DBA_FREE_SPACE
         GROUP BY TABLESPACE_NAME) B
    WHERE A.TABLESPACE_NAME = B.TABLESPACE_NAME(+)
    ORDER BY 5
        """

    cursor = conn.cursor()
    cursor.execute(space_usage)
    usage = cursor.fetchall()
    usage_title = [i[0] for i in cursor.description]
    usage_data = pd.DataFrame(np.array(usage), columns=usage_title)
    print(usage_data)
    cursor.close()

    for index, row in usage_data.iterrows():

        # print(row['REDO'])
        # for ind in range(len(title)):
        # from datetime import datetime
        # snap_date=datetime.strptime(row['SNAP_DATE'], '%y/%m/%d').strftime('%Y-%m-%d')

        models.spaceusage.objects.create(
            collect_time = datetime.datetime.strftime(datetime.datetime.now(), '%Y-%m-%d %H'),
            #data_time=datetime.datetime.strptime(row['DATA_TIME'], '%Y-%m-%d %H'),
            tablespace_name=row['TABLESPACE_NAME'],
            total=row['TOTAL'],
            free=row['FREE'],
            used=row['USED'],
            percent=row['PERCENT']
        )

@periodic_task(run_every = 3600,name = "collect_system_metric_period")
def collect_system_metric_period():

    conn = database_method.initial_connect('system', 'oracle', 'trn')
    conn = conn.create_conn()
    system_metric = """
    select begin_time,end_time,metric_name,metric_unit,average,standard_deviation,sum_squares from dba_hist_sysmetric_summary where to_char(begin_time,'yyyy-mm-dd hh24') = :var
    """
    try:
        cursor = conn.cursor()
        var = datetime.datetime.strftime(datetime.datetime.now()+datetime.timedelta(hours=-2), '%Y-%m-%d %H')
        print(var)
        cursor.execute(system_metric,var=var)
        collect_system_metric = cursor.fetchall()
        collect_system_metric_title = [i[0] for i in cursor.description]
        collect_system_metric_data = pd.DataFrame(np.array(collect_system_metric), columns=collect_system_metric_title)
        print(collect_system_metric_data)
        cursor.close()

        for index, row in collect_system_metric_data.iterrows():
            models.system_metric_period.objects.create(
                collect_time= datetime.datetime.strftime(datetime.datetime.now(), '%Y-%m-%d %H'),
                begin_time= row['BEGIN_TIME'],
                end_time= row['END_TIME'],
                metric_name= row['METRIC_NAME'],
                data_value = row['METRIC_UNIT'],
                metric_average = row['AVERAGE'],
                metric_standard = row['STANDARD_DEVIATION'],
                metric_squares = row['SUM_SQUARES']
            )
    except Exception as msg:
        print(msg)

@periodic_task(run_every = 3600,name = "tablespace_channge")
def tablespace_channge():

    conn = database_method.initial_connect('system', 'oracle', 'trn')
    conn = conn.create_conn()

    tablespace_change = """
                    select rtime,tablespace_name,tablespace_usedsize_kb,tablespace_size_kb,diff_kb from 
                    (with tmp as
                    (select  rtime,tablespace_name,
                    sum(tablespace_usedsize_kb) tablespace_usedsize_kb,
                    sum(tablespace_size_kb) tablespace_size_kb
                    from (select rtime,
                    e.tablespace_id,f.tablespace_name as tablespace_name,
                    (e.tablespace_usedsize) * (f.block_size) / 1024 tablespace_usedsize_kb,
                    (e.tablespace_size) * (f.block_size) / 1024 tablespace_size_kb
                    from dba_hist_tbspc_space_usage e,
                    dba_tablespaces f,
                    v$tablespace g
                    where e.tablespace_id = g.TS#
                    and f.tablespace_name = g.NAME
                    and f.tablespace_name in ('SOE')
                    )
                    group by rtime,tablespace_name)
                    select tmp.rtime,tmp.tablespace_name,
                    tablespace_usedsize_kb,
                    tablespace_size_kb,
                    (tablespace_usedsize_kb -
                    LAG(tablespace_usedsize_kb, 1, NULL) OVER(ORDER BY tmp.rtime)) AS DIFF_KB
                    from tmp,
                    (select rtime rtime,tablespace_name
                    from tmp
                    group by rtime,tablespace_name) t2
                    where t2.rtime = tmp.rtime and t2.tablespace_name=tmp.tablespace_name
                    order by rtime)
                    where to_char(to_date(rtime,'mm/dd/yyyy hh24:mi:ss'),'yyyy-mm-dd hh24') = :var
                """
    try:
        cursor = conn.cursor()
        var = datetime.datetime.strftime(datetime.datetime.now(), '%Y-%m-%d %H')
        cursor.execute(tablespace_change,var=var)
        tablespace_change_metric = cursor.fetchall()
        tablespace_change_metric_title = [i[0] for i in cursor.description]
        tablespace__change_metric_data = pd.DataFrame(np.array(tablespace_change_metric), columns=tablespace_change_metric_title)
        print(tablespace__change_metric_data)
        cursor.close()

        for index, row in tablespace__change_metric_data.iterrows():

            # print(row['REDO'])
            # for ind in range(len(title)):
            # from datetime import datetime
            # snap_date=datetime.strptime(row['SNAP_DATE'], '%y/%m/%d').strftime('%Y-%m-%d')

            models.spacechange.objects.create(
                collect_time = row['RTIME'],
                #data_time=datetime.datetime.strptime(row['DATA_TIME'], '%Y-%m-%d %H'),
                tablespace_name=row['TABLESPACE_NAME'],
                tablespace_usedsize_kb=row['TABLESPACE_USEDSIZE_KB'],
                tablespace_size_kb=row['TABLESPACE_SIZE_KB'],
                DIFF_KB=row['DIFF_KB']
            )
    except Exception as msg:
        print(msg)

@periodic_task(run_every = 720,name = "get_mysql_metric")
def get_mysql_metric():

    metric_ = fetch_mysql_metric()
    # #before_time = time.strftime("%d_%m_%Y-%H_%M_%S")
    begin_time = datetime.datetime.now()
    metric_.get_mysql_metric("before",time.strftime("%d_%m_%Y-%H_%M_%S"))
    os.chdir("/u01/tpcc-mysql")


    #随机生成负载
    pereiod = random.randint(100, 500)
    warehouse_number = random.randint(1,100)
    conn_number = random.randint(20,50)

    os.system("./tpcc_start -h101.132.149.24 -P3306 -d tpccdb -u root -p Edwin703 -w "+str(warehouse_number)+" -c "+str(conn_number)+" -r 50 -l "+str(pereiod))

    end_time = datetime.datetime.now()
    metric_ = fetch_mysql_metric()
    metric_.get_mysql_metric("after",time.strftime("%d_%m_%Y-%H_%M_%S"))

    os.chdir("/oracle_predict/" + time.strftime("%d_%m_%Y"))
    #filename = "data" + time.strftime("%d_%m_%Y-%H_%M_%S") + ".summary"
    filename = "summary.json"
    res_ = {
               "start_time": int(time.mktime(begin_time.timetuple())),
               "end_time": int(time.mktime(end_time.timetuple())),
               "observation_time": (end_time-begin_time).seconds,
               "database_type": "mysql",
               "database_version": "5.7",
               "workload_name": "wk1"
            }
    with open(filename, 'w') as file_obj:
        JSONUtil.dump(res_, file_obj)
    #return render(request, "./node_modules/gentelella/production/fetch_metric_data_ok.html",{'title':"系统负载数据收集完成","time":time.strftime("%d_%m_%Y-%H_%M_%S")})

    os.chdir("/oracle_predict/" + time.strftime("%d_%m_%Y"))
    os.system("python upload.py " + "/oracle_predict/" + time.strftime(
        "%d_%m_%Y") + " WL9FR3445C1UR9UCFA55 http://106.15.227.92:8080/new_result/")