# -*- coding: utf-8 -*-
# @Time    : 2019/4/11 10:07 PM
# @Author  : Edwin
# @File    : test.py
# @Software: PyCharm


from time_series_detector import detect
import pandas as pd
import warnings
import matplotlib.pyplot as plt
from ora_dual import models
import os,time,datetime
import pandas

warnings.filterwarnings('ignore')
pd.set_option('display.max_columns', 1000)
pd.set_option('display.width', 1000)
pd.set_option('display.max_colwidth', 1000)
pd.set_option('display.max_rows',1000)


#
# spacechange_metric_data = pd.read_csv('spacechange_metric_05_04_2019.csv')
# spacechange_metric_data = spacechange_metric_data[
#         ['begin_time','DB Block Changes Per Sec','DIFF_KB']]
#
# total_block_change = spacechange_metric_data[['begin_time','DB Block Changes Per Sec']]
# #print(total_block_change)
# total_block_change.plot(x='begin_time',y='DB Block Changes Per Sec')
# plt.show()
#
# # detect_data = spacechange_metric_data.loc[spacechange_metric_data['begin_time']=='2019-04-03  09']
# # detect_data = list(detect_data['DB Block Changes Per Sec'])
# # detect_data = [str(int(x)) for x in detect_data]detect_data
# detect_data = ["1000"]
# detect_data = ",".join(detect_data)
#
#
# datac = list(spacechange_metric_data['DB Block Changes Per Sec'][0:360])
# datac.append(detect_data)
# datac = [str(int(x)) for x in datac]
# datac = ",".join(datac)
# #print("datac is",datac)
# # print(len(datac.split(',')))
# #
# datab = list(spacechange_metric_data['DB Block Changes Per Sec'][361:721])
# datab.append(detect_data)
# datab = [str(int(x)) for x in datab]
# datab = ",".join(datab)
# #print("datab is",datab)
# #
# dataa = list(spacechange_metric_data['DB Block Changes Per Sec'][722:902])
# dataa.append(detect_data)
# dataa = [str(int(x)) for x in dataa]
# dataa = ",".join(dataa)
# #print("dataa is",dataa)
# #
# #
# detect_obj = detect.Detect()
# data = {"window":180,
#         "dataC":datac,
#         "dataB":datab,
#         "dataA":dataa
#         }
# # combined_data = data["dataC"] + "," + data["dataB"] + "," + data["dataA"]
# # time_series = map(int, combined_data.split(','))
# print(detect_obj.value_predict(data))

load_profile_per_hour = list(models.system_metric_period.objects.values('begin_time', 'metric_name', 'metric_average').distinct().order_by(
            'begin_time'))

# load_profile_columns = list(models.system_metric_period.objects.values('metric_name').distinct())
# load_porfile_time = list(models.system_metric_period.objects.values('begin_time').distinct())

load_profile_per_hour = pd.DataFrame(load_profile_per_hour)

load_profile_per_hour_out = load_profile_per_hour.pivot(index='begin_time', columns='metric_name',values='metric_average')


load_profile_per_hour_out.to_csv('spacechange_metric_' + time.strftime("%d_%m_%Y") + '.csv', index=True, header=True,
                                     na_rep=0)
load_profile_per_hour_out = pandas.read_csv('spacechange_metric_' + time.strftime("%d_%m_%Y") + '.csv')

load_profile_per_hour_out['begin_time'] = load_profile_per_hour_out['begin_time'].apply(
        lambda x: datetime.datetime.strftime(datetime.datetime.strptime(x, '%Y-%m-%d  %H:%M:%S'),'%Y-%m-%d  %H'))


