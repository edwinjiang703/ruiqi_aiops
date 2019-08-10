from django.shortcuts import render
from django.shortcuts import HttpResponse
from django.shortcuts import redirect
from ora_dual import tasks
from celery.result import AsyncResult
# Create your views here.
from ora_dual import models
from ora_dual import database_method
import pandas as pd
import seaborn as sns
import numpy as np
from django.apps import apps
from fetch_knob_metric.fetch_mysql_knob_data import fetch_mysql_knob
from fetch_knob_metric.fetch_mysql_metric_data import fetch_mysql_metric
from fetch_knob_metric.fetch_oracle_knob_data import fetch_oracle_knob_data
from fetch_knob_metric.fetch_oracle_metric_data import fetch_oracle_metic_data
from fetch_knob_metric.upload_oracle_knobcatalog import upload_ora_knob
from fetch_knob_metric.upload_oracle_metriccatalog import upload_ora_metric
from django.contrib.auth.decorators import login_required
from django.urls import reverse, reverse_lazy
from matplotlib import pyplot as plt
from fetch_knob_metric.JSONUtile import JSONUtil
import os,time,datetime,random
from time_series_detector import detect
import pandas as pd
import logging
USER_LIST = []

load_profile_per_hour = []

def index(request):
    #return HttpResponse("hello oracle predict")
    #return render(request,"index.html")
    res = tasks.add.delay(5,10)
    print("res:",res)
    # return HttpResponse(res.task_id)

    #res = tasks.run_cmd("df -h")
    #print("res:",res)
    return HttpResponse(res)

#跟胡task id获取结果
def task_res(request):
    result = AsyncResult(id="514ecf14-6b06-40e8-9111-196c2f8ecee1")
    return HttpResponse(result.get())


def login(request):
    # f = open('templates/login.html','r',encoding='utf-8')
    # logstr = f.read()
    # return HttpResponse(logstr)

    error_msg =''
    if request.method == 'POST':
        username = request.POST.get('username',None)
        password = request.POST.get('passwd',None)
        print(username)
        if username == 'root' and password == '12345':
            return redirect("/ora_dual/home")
        else:
            error_msg ="username or password is not correct"


    return render(request,'welcome.html',{"error_msg":error_msg})

def sapreport(request):
    return render(request, "./node_modules/gentelella/production/sap_template.html")

def home(request):
    # return render(request,'home.html')
    if request.method == 'POST':
        username = request.POST.get('username',None)
        age = request.POST.get("age",None)
        gender = request.POST.get("gender",None)
        usergroup_id = request.POST.get("group",None)
        models.UserInfo.objects.create(username=username, age=age, gender=gender,usergroup_id=usergroup_id)
    USER_LIST = list(models.UserInfo.objects.all())
    return render(request,"./node_modules/gentelella/production/home.html",{"userlist":USER_LIST})

def delete(request):
    nid = request.GET.get('nid')
    if nid:
        models.UserInfo.objects.filter(id=nid).delete()
    return redirect("/ora_dual/home")


def detail(request,*args,**kwargs):
    nid = request.GET.get('nid')
    #nid = kwargs['nid']
    #
    # if id:
    #     userinfo = USER_LIST[int(id)]
    #     return render(request,'detail.html',{'userinfo':userinfo})
    if nid:
        userinfo = models.UserInfo.objects.filter(id=nid)
        print(userinfo)
        return render(request,'detail.html',{'user_info':userinfo})

from django.views import View

class TestCBV(View):
    def get(self,request):
        print(request.method)
        return render(request,'test.html')

    def post(self,request):
        print(request.method)
        return render(request,'test.html')


#ORM 操作 增删改查

def ormtest(request):

    #插入数据的三种方式：
    # 1
    # models.UserInfo.objects.create(username='ormtest',password='ormpwd')

    # 2
    # dictdata = {'username':'test','password':'passwd'}
    # models.UserInfo.objects.create(**dictdata)

    # 3
    # obj = models.UserInfo(username='testobj',password='passwdobj')
    # obj.save()

    #查询数据
    # res = models.UserInfo.objects.all()
    #
    # for info in res:
    #     print(info.username)
    #     print(info.password)

    # res = models.UserInfo.objects.filter(username = 'test')
    # for info in res:
    #     print(info.username)
    # print(res)

    # #修改数据
    # models.UserInfo.objects.filter(username='test').update(password='123456')
    #
    # #删除数据
    # models.UserInfo.objects.filter(username='test').delete()


    return HttpResponse("ok")


def get_load_profile_data(request):

    # if request.method == 'POST':
    conn = database_method.initial_connect('system','oracle','trn')
    conn = conn.create_conn()
    load_profile = """select
           s.instance_number,
           s.snap_date,
           decode(s.redosize, null, '--shutdown or end--', s.currtime) "TIME",
           to_char(round(s.seconds/60,2)) "elapse(min)",
           round(t.db_time / 1000000 / 60, 2) "DB time(min)",
           s.redosize redo,
           round(s.redosize / s.seconds, 2) "redo/s",
           s.logicalreads logical,
           round(s.logicalreads / s.seconds, 2) "logical/s",
           physicalreads physical,
           round(s.physicalreads / s.seconds, 2) "phy/s",
           s.executes execs,
           round(s.executes / s.seconds, 2) "execs/s",
           s.parse,
           round(s.parse / s.seconds, 2) "parse/s",
           s.hardparse,
           round(s.hardparse / s.seconds, 2) "hardparse/s",
           s.transactions trans,
           round(s.transactions / s.seconds, 2) "trans/s"
      from (select curr_redo - last_redo redosize,
                   curr_logicalreads - last_logicalreads logicalreads,
                   curr_physicalreads - last_physicalreads physicalreads,
                   curr_executes - last_executes executes,
                   curr_parse - last_parse parse,
                   curr_hardparse - last_hardparse hardparse,
                   curr_transactions - last_transactions transactions,
                   round(((currtime + 0) - (lasttime + 0)) * 3600 * 24, 0) seconds,
                   to_char(currtime, 'yyyy-mm-dd') snap_date,
                   to_char(currtime, 'hh24:mi') currtime,
                   currsnap_id endsnap_id,
                   to_char(startup_time, 'yyyy-mm-dd hh24:mi:ss') startup_time,
                   instance_number
              from (select a.redo last_redo,
                           a.instance_number instance_number,
                           a.logicalreads last_logicalreads,
                           a.physicalreads last_physicalreads,
                           a.executes last_executes,
                           a.parse last_parse,
                           a.hardparse last_hardparse,
                           a.transactions last_transactions,
                           lead(a.redo, 1, null) over(partition by b.startup_time order by b.end_interval_time) curr_redo,
                           lead(a.logicalreads, 1, null) over(partition by b.startup_time order by b.end_interval_time) curr_logicalreads,
                           lead(a.physicalreads, 1, null) over(partition by b.startup_time order by b.end_interval_time) curr_physicalreads,
                           lead(a.executes, 1, null) over(partition by b.startup_time order by b.end_interval_time) curr_executes,
                           lead(a.parse, 1, null) over(partition by b.startup_time order by b.end_interval_time) curr_parse,
                           lead(a.hardparse, 1, null) over(partition by b.startup_time order by b.end_interval_time) curr_hardparse,
                           lead(a.transactions, 1, null) over(partition by b.startup_time order by b.end_interval_time) curr_transactions,
                           b.end_interval_time lasttime,
                           lead(b.end_interval_time, 1, null) over(partition by b.startup_time order by b.end_interval_time) currtime,
                           lead(b.snap_id, 1, null) over(partition by b.startup_time order by b.end_interval_time) currsnap_id,
                           b.startup_time
                      from (select snap_id,
                                   dbid,
                                   instance_number,
                                   sum(decode(stat_name, 'redo size', value, 0)) redo,
                                   sum(decode(stat_name,
                                              'session logical reads',
                                              value,
                                              0)) logicalreads,
                                   sum(decode(stat_name,
                                              'physical reads',
                                              value,
                                              0)) physicalreads,
                                   sum(decode(stat_name, 'execute count', value, 0)) executes,
                                   sum(decode(stat_name,
                                              'parse count (total)',
                                              value,
                                              0)) parse,
                                   sum(decode(stat_name,
                                              'parse count (hard)',
                                              value,
                                              0)) hardparse,
                                   sum(decode(stat_name,
                                              'user rollbacks',
                                              value,
                                              'user commits',
                                              value,
                                              0)) transactions
                              from dba_hist_sysstat
                             where stat_name in
                                   ('redo size',
                                    'session logical reads',
                                    'physical reads',
                                    'execute count',
                                    'user rollbacks',
                                    'user commits',
                                    'parse count (hard)',
                                    'parse count (total)')
                             group by snap_id, dbid, instance_number) a,
                           dba_hist_snapshot b
                     where a.snap_id = b.snap_id
                       and a.dbid = b.dbid
                       and a.instance_number = b.instance_number
                       --and a.dbid = &&spool_dbid
                       --and a.instance_number = &&spool_inst_num
                     order by end_interval_time)) s,
           (select lead(a.value, 1, null) over(partition by b.startup_time order by b.end_interval_time) - a.value db_time,
                   lead(b.snap_id, 1, null) over(partition by b.startup_time order by b.end_interval_time) endsnap_id,
                   a.instance_number instance_number
              from dba_hist_sys_time_model a, dba_hist_snapshot b
             where a.snap_id = b.snap_id
               and a.dbid = b.dbid
               and a.instance_number = b.instance_number
               and a.stat_name = 'DB time'
               --and a.dbid = &&spool_dbid
               --and a.instance_number = &&spool_inst_num
               ) t
     where s.endsnap_id = t.endsnap_id
     and s.instance_number = t.instance_number
     order by  s.snap_date desc ,time asc"""

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



    # 执行查询 语句
    cursor = conn.cursor()
    cursor.execute(load_profile)
    load_metric = cursor.fetchall()

    title = [i[0] for i in cursor.description]
    load_profile_per_hour = pd.DataFrame(np.array(load_metric), columns=title)
    #print(load_profile_per_hour.ix[0])
    cursor.close()

    cursor = conn.cursor()
    cursor.execute(space_usage)
    usage = cursor.fetchall()

    usage_title = [i[0] for i in cursor.description]
    usage_data  = pd.DataFrame(np.array(usage), columns=usage_title)
    print(usage_data)
    cursor.close()




    for index,row in usage_data.iterrows():
        print(row['PERCENT'])
        # print(row['REDO'])
        #for ind in range(len(title)):
        # from datetime import datetime
        # snap_date=datetime.strptime(row['SNAP_DATE'], '%y/%m/%d').strftime('%Y-%m-%d')
        models.spaceusage.objects.create(
                                              collect_time=row['DATA_TIME'],
                                              tablespace_name= row['TABLESPACE_NAME'],
                                              total = row['TOTAL'],
                                              free = row['FREE'],
                                              used = row['USED'],
                                              percent= row['PERCENT']
                                              )

    for index, row in load_profile_per_hour.iterrows():
        # print(row['REDO'])c
        # for ind in range(len(title)):

        # from datetime import datetime
        # snap_date = datetime.strptime(row['SNAP_DATE'], '%y/%m/%d ').strftime('%Y-%m-%d')
        models.loadmetric_hour.objects.create(instance_number=int(row['INSTANCE_NUMBER']),
                                              snap_date=row['SNAP_DATE'],
                                              time=row['TIME'],
                                              elapse_min=row['elapse(min)'],
                                              dbtime_min=row['DB time(min)'],
                                              redo=row['REDO'],
                                              redo_second=row['redo/s'],
                                              logical=row['LOGICAL'],
                                              logical_second=row['logical/s'],
                                              physical=row['PHYSICAL'],
                                              physical_second=row['phy/s'],
                                              execs=row['EXECS'],
                                              execs_second=row['execs/s'],
                                              parse=row['PARSE'],
                                              parse_second=row['parse/s'],
                                              hardware=row['HARDPARSE'],
                                              harware_second=row['hardparse/s'],
                                              trans=row['TRANS'],
                                              trans_second=row['trans/s']
                                              )
            #

    load_profile_per_hour = list(models.loadmetric_hour.objects.all())
    load_profile_obj = apps.get_model('ora_dual','loadmetric_hour')
    load_profile_field = load_profile_obj._meta.fields
    title = []
    for ind in range(len(load_profile_field)):
        title.append(load_profile_field[ind].name)

    return render(request, "load_profile_per_hour.html", {"title":title,"load_metric": load_profile_per_hour})


def sel_metric_data(request):
    snap_date = list(models.loadmetric_hour.objects.values("snap_date").distinct())
    return render(request, "./node_modules/gentelella/production/sel_metric_data.html", {'snap_date': snap_date})


def get_mysql_knob(request):
    knob_ = fetch_mysql_knob()
    knob_.get_mysql_knob()
    return render(request, "./node_modules/gentelella/production/fetch_metric_data_ok.html",{'title': "参数收集完成","time":time.strftime("%d_%m_%Y-%H_%M_%S")})

def get_mysql_metric(request):

    data_time = time.strftime("%d_%m_%Y")

    metric_ = fetch_mysql_metric()
    # #before_time = time.strftime("%d_%m_%Y-%H_%M_%S")
    begin_time = datetime.datetime.now()
    metric_.get_mysql_metric("before",time.strftime("%d_%m_%Y-%H_%M_%S"))
    os.chdir("/u01/tpcc-mysql")

    #随机生成负载
    pereiod = random.randint(100, 500)   # 运行多长时间
    warehouse_number = random.randint(1,100)
    conn_number = random.randint(20,50)

    #TPCC-Mysql参数用法：
    # -h
    # server_host: 服务器名
    # -P
    # port: 端口号，默认为3306
    # -d
    # database_name: 数据库名
    # -u
    # mysql_user: 用户名
    # -p
    # mysql_password: 密码
    # -w
    # warehouses: 仓库的数量
    # -c
    # connections: 线程数，默认为1
    # -r
    # warmup_time: 热身时间，单位: s，默认为10s, 热身是为了将数据加载到内存。
    # -l
    # running_time: 测试时间，单位: s，默认为20s
    # -i
    # report_interval: 指定生成报告间隔时长
    # -f
    # report_file: 测试结果输出文件

    os.system("./tpcc_start -h101.132.149.24 -P3306 -d tpccdb -u root -p Edwin703 -w "+str(warehouse_number)+" -c "+str(conn_number)+" -r 50 -l "+str(pereiod))
    #os.system("./tpcc_start -h101.132.149.24 -P3306 -d tpccdb -u root -p Edwin703 -w 10 -c 10 -r 10 -l 50")

    end_time = datetime.datetime.now()
    metric_ = fetch_mysql_metric()
    metric_.get_mysql_metric("after",time.strftime("%d_%m_%Y-%H_%M_%S"))

    os.chdir("/oracle_predict/" + str(data_time))
    #filename = "data" + time.strftime("%d_%m_%Y-%H_%M_%S") + ".summary"
    filename = "summary.json"
    res_ = {
               "start_time": int(time.mktime(begin_time.timetuple())),
               "end_time": int(time.mktime(end_time.timetuple())),
               "observation_time": (end_time-begin_time).seconds,
               "database_type": "mysql",
               "database_version": "5.7",
               "workload_name": "oltp_1"
            }
    with open(filename, 'w') as file_obj:
        JSONUtil.dump(res_, file_obj)
    #return render(request, "./node_modules/gentelella/production/fetch_metric_data_ok.html",{'title':"系统负载数据收集完成","time":time.strftime("%d_%m_%Y-%H_%M_%S")})

    os.chdir("/oracle_predict/upload_data")
    os.system("python upload.py " + "/oracle_predict/" + str(data_time) + " WL9FR3445C1UR9UCFA55 http://106.15.227.92:8080/new_result/")
    return render(request, "./node_modules/gentelella/production/fetch_metric_data_ok.html",
                  {'title': "Mysql负载数据上传完成", "time": time.strftime("%d_%m_%Y-%H_%M_%S")})

def upload_data(request):
    os.chdir("/oracle_predict/"+time.strftime("%d_%m_%Y"))
    os.system("python upload.py "+"/oracle_predict/"+time.strftime("%d_%m_%Y")+" WL9FR3445C1UR9UCFA55 http://106.15.227.92:8080/new_result/")
    return render(request, "./node_modules/gentelella/production/fetch_metric_data_ok.html",
                  {'title': "数据上传完成", "time": time.strftime("%d_%m_%Y-%H_%M_%S")})



def get_ora_parameter_metadata(request):
    knob_ = fetch_oracle_knob_data()
    knob_.collect_parameter_data()
    return render(request, "./node_modules/gentelella/production/fetch_metric_data_ok.html",
                  {'title': "Mysqk参数元数据收集完成", "time": time.strftime("%d_%m_%Y-%H_%M_%S")})


def upload_ora_parameter_metadata(request):

    catalog_ = upload_ora_knob()
    catalog_.upload_parm_metadata()
    return render(request, "./node_modules/gentelella/production/fetch_metric_data_ok.html",
              {'title': "Oracle参数元数据上传完成", "time": time.strftime("%d_%m_%Y-%H_%M_%S")})


def get_oracle_metric(request):
    data_time = time.strftime("%d_%m_%Y")
    begin_time = datetime.datetime.now()
    metric_ = fetch_oracle_metic_data()
    metric_.collect_metic_data('before')

    #oracle swingbench 测试
    # os.chdir('/swingbench/bin')
    # os.system("./charbench -c /swingbench/configs/SOE_Server_Side_V2.xml -cpuloc 47.101.133.179 -cpuuser oracle -cpupass oracle -v 'users,cpu,disk,tpm,tps,resp' -rt 00:03 > oracle_swingbench_test.log 2>&1 &")   #rt 测试运行多长时间

    time.sleep(300)

    end_time = datetime.datetime.now()
    metric_ = fetch_oracle_metic_data()
    metric_.collect_metic_data('after')

    os.chdir("/oracle_predict/oracle_data/" + str(data_time))
    # filename = "data" + time.strftime("%d_%m_%Y-%H_%M_%S") + ".summary"
    filename = "summary.json"
    res_ = {
        "start_time": int(time.mktime(begin_time.timetuple())),
        "end_time": int(time.mktime(end_time.timetuple())),
        "observation_time": (end_time - begin_time).seconds,
        "database_type": "oracle",
        "database_version": "12",
        "workload_name": "oracle_test"
    }
    with open(filename, 'w') as file_obj:
        JSONUtil.dump(res_, file_obj)
    # return render(request, "./node_modules/gentelella/production/fetch_metric_data_ok.html",{'title':"系统负载数据收集完成","time":time.strftime("%d_%m_%Y-%H_%M_%S")})

    os.chdir("/oracle_predict/upload_data")
    os.system("python upload.py " + "/oracle_predict/oracle_data/" + str(
        data_time) + " 5V1TL8SWYJFQASKJCSCS http://106.15.227.92:8080/new_result/")
    return render(request, "./node_modules/gentelella/production/fetch_metric_data_ok.html",
                  {'title': "Oracle负载数据上传完成", "time": time.strftime("%d_%m_%Y-%H_%M_%S")})


def upload_oracle_metric_metadata(request):

    metric_ = upload_ora_metric()
    metric_.upload_metric_metadata()
    return render(request, "./node_modules/gentelella/production/fetch_metric_data_ok.html",
                  {'title': "Oracle负载元数据上传完成", "time": time.strftime("%d_%m_%Y-%H_%M_%S")})

def detect_metis_value(request):


    load_profile_per_hour = list(
        models.system_metric_period.objects.values('begin_time', 'metric_name', 'metric_average').distinct().order_by(
            'begin_time'))

    # load_profile_columns = list(models.system_metric_period.objects.values('metric_name').distinct())
    # load_porfile_time = list(models.system_metric_periodvalues('begin_time').distinct())

    load_profile_per_hour = pd.DataFrame(load_profile_per_hour)


    load_profile_per_hour_out = load_profile_per_hour.pivot(index='begin_time', columns='metric_name',
                                                            values='metric_average')

    load_profile_per_hour_out.to_csv('spacechange_metric_' + time.strftime("%d_%m_%Y") + '.csv', index=True,
                                     header=True,na_rep=0)
    load_profile_per_hour_out = pd.read_csv('spacechange_metric_' + time.strftime("%d_%m_%Y") + '.csv')


    load_profile_per_hour_out['begin_time'] = load_profile_per_hour_out['begin_time'].apply(
        lambda x: datetime.datetime.strftime(datetime.datetime.strptime(x, '%Y-%m-%d  %H:%M:%S'), '%Y-%m-%d  %H'))

    change_data_ = (row for row in load_profile_per_hour_out[['begin_time','DB Block Changes Per Sec']].iterrows())

    return render(request, "./node_modules/gentelella/production/fetch_metric_data_ok.html",
                  {'title': "file created", "time": time.strftime("%d_%m_%Y-%H_%M_%S")})

    # detect_data = ["1000"]
    # detect_data = ",".join(detect_data)
    # #
    # datac = list(change_data_['DB Block Changes Per Sec'][0:360])
    # datac.append(detect_data)
    # datac = [str(int(x)) for x in datac]
    # datac = ",".join(datac)
    #
    # datab = list(change_data_['DB Block Changes Per Sec'][361:721])
    # datab.append(detect_data)
    # datab = [str(int(x)) for x in datab]
    # datab = ",".join(datab)
    #
    # dataa = list(change_data_['DB Block Changes Per Sec'][722:902])
    # dataa.append(detect_data)
    # dataa = [str(int(x)) for x in dataa]
    # dataa = ",".join(dataa)
    # #
    # detect_obj = detect.Detect()
    # data = {"window": 180,
    #         "dataC": datac,
    #         "dataB": datab,
    #         "dataA": dataa
    #         }
    #
    # if detect_obj.value_predict(data)[1]["ret"] == 1:
    #
    #     return render(request, "./node_modules/gentelella/production/fetch_metric_data_ok.html",
    #               {'title': "检测结果为正常值", "time": time.strftime("%d_%m_%Y-%H_%M_%S")})
    # else:
    #     return render(request, "./node_modules/gentelella/production/fetch_metric_data_ok.html",
    #                   {'title': "检测结果为异常值", "time": time.strftime("%d_%m_%Y-%H_%M_%S")})