#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2019/1/25
# @Author  : Edwin
# @Version : Python 3.6
# @File    : space_predict_122.py

import cx_Oracle
import numpy as np
import sklearn
import pandas as pd
import datetime
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler,MinMaxScaler
from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn.linear_model import Lasso,Ridge
from pyecharts import Line
import pickle as pk
from ora_dual import models
from django.http import HttpResponse
from django.template import loader
import pandas
import time
pd.set_option('display.max_columns', 1000)
pd.set_option('display.width', 1000)
pd.set_option('display.max_colwidth', 1000)
pd.set_option('display.max_rows',1000)


#获得metric和SOE表空间变化数据
def process_space_metric_data(request):

    #load_profile_per_hour = list(models.system_metric_period.objects.values('begin_time','metric_name','metric_average').filter(begin_time__range=["2019-01-23 20:00","2019-01-23 20:59"]))
    load_profile_per_hour = list(models.system_metric_period.objects.values('begin_time', 'metric_name', 'metric_average').distinct().order_by(
            'begin_time'))

    # load_profile_columns = list(models.system_metric_period.objects.values('metric_name').distinct())
    #
    # load_profile_index = list(models.system_metric_period.objects.values('begin_time').distinct())
    #
    # columns = []
    # for idx in range(len(load_profile_columns)):
    #     columns.append(load_profile_columns[idx]['metric_name'])
    #
    # index = []
    # for idx in range(len(load_profile_index)):
    #     index.append(load_profile_index[idx]['begin_time'])
    # index_frame = pd.Index(data=index, name="begin_time")
    #
    # data =[]
    # for idx in range(len(index)):
    #     for idx_col in range(len(columns)):
    #         for idx_val in range(len(load_profile_per_hour)):
    #             if load_profile_per_hour[idx_val]['begin_time'] == index[idx] and load_profile_per_hour[idx_val]['metric_name'] == columns[idx_col]:
    #                 data.append(load_profile_per_hour[idx_val]['metric_average'])
    #             else:
    #                 data.append(0)
    #
    #
    # load_profile = pd.DataFrame(data,columns=columns,index=index_frame)
    # print(load_profile.head(10))

    #print(load_profile_per_hour)

    # df = pd.DataFrame()
    #
    # for idx in range(len(load_profile_per_hour)):
    #     tmp = pd.DataFrame(load_profile_per_hour[idx],index = [0])
    #     print(tmp)
    #     #df.append(tmp,ignore_index=True)
    #
    # print(df.head(10))

    load_profile_per_hour = pd.DataFrame(load_profile_per_hour)
    #
    # load_profile_per_hour.index.name='begin_time'
    # load_profile_per_hour.columns.name='metric_name'

    df = pd.melt(load_profile_per_hour, id_vars=['begin_time', 'metric_name'],
                 value_vars=list(load_profile_per_hour.columns)[2:],
                 var_name='begin_time', value_name='metric_average')
    # df['date'] = df['begin_time'].str[1:].astype('float')
    # df['date'] = df[['year', 'month', 'date']].apply(
    #     lambda row: '{:4d}-{:02d}-{:02d}'.format(*row),
    #     axis=1)
    #df = df.loc[df['value'] != '---', ['id', 'date', 'element', 'value']]
    df = df.set_index(['begin_time', 'metric_name'])
    df = df.unstack()
    df.columns = list(df.columns.get_level_values('metric_name'))
    df = df.reset_index()

    print(df.head(10))



    #load_profile_per_hour.drop_duplicates(subset=['begin_time'], keep='first', inplace=True)
    load_profile_per_hour_out = load_profile_per_hour.pivot(index='begin_time', columns='metric_name',
                                                            values='metric_average')
    #sprint(load_profile_per_hour_out)

    load_profile_per_hour_out.to_csv('spacechange_metric_' + time.strftime("%d_%m_%Y") + '.csv', index=True, header=True,
                                     na_rep=0)

    load_profile_per_hour_out = pandas.read_csv('spacechange_metric_' + time.strftime("%d_%m_%Y") + '.csv')

    load_profile_per_hour_out['begin_time'] = load_profile_per_hour_out['begin_time'].apply(
        lambda x: datetime.datetime.strftime(datetime.datetime.strptime(x, '%Y-%m-%d  %H:%M:%S'), '%Y-%m-%d  %H'))


    # gt == == 大于
    # gte == = 大于等于
    # lt == == = 小于
    # lte == == 小于等于
    #spacechange_data = list(models.spacechange.objects.values().filter(collect_time__lt=datetime.datetime.strftime(datetime.datetime.strptime('01/24/2019 00:00:00','%m/%d/%Y  %H:%M:%S'),'%Y-%m-%d  %H')).all())
    spacechange_data = list(models.spacechange.objects.values().all())
    spacechange_data = pd.DataFrame(spacechange_data)
    spacechange_data['collect_time'] = spacechange_data['collect_time'].apply(lambda x:datetime.datetime.strftime(datetime.datetime.strptime(x,'%m/%d/%Y  %H:%M:%S'),'%Y-%m-%d  %H'))
    spacechange_data = spacechange_data[spacechange_data.collect_time >= '2019-01-22 00']
    spacechange_data['begin_time'] = spacechange_data['collect_time']

    #spacechange_data.drop(['collect_time'],inplace=True)

    # load_profile_per_hour['diffkb'] = spacechange_data['DIFF_KB'].apply(
    #     lambda x: spacechange_data['DIFF_KB'] if spacechange_data['collect_time'] == load_profile_per_hour[
    #         'begin_time'] else 0)

    #print(spacechange_data)
    #print(load_profile_per_hour)

    #几种dataframe行转列的方法
    #load_profile_per_hour_out = load_profile_per_hour.set_index(['begin_time','metric_name']).unstack()
    # load_profile_per_hour_out = load_profile_per_hour.groupby(['begin_time','metric_name'])['metric_average'].apply(float).unstack()
    #load_profile_per_hour_out.columns = load_profile_per_hour_out.columns.map('{0[1]}'.format)

    #print(load_profile_per_hour_out.ix[1:2])
    loadprofile_spacechange = pandas.merge(load_profile_per_hour_out,spacechange_data,on=['begin_time'],how='inner')
    loadprofile_spacechange.drop(['collect_time','id','tablespace_name','tablespace_size_kb','tablespace_usedsize_kb'],inplace=True,axis = 1)
    loadprofile_spacechange.to_csv('spacechange_metric_' + time.strftime("%d_%m_%Y") + '.csv', index=True, header=True,
                                     na_rep=0)
    #print(loadprofile_spacechange.shape[0],loadprofile_spacechange.shape[1])
    #163个特征
    return HttpResponse(datetime.datetime.now().strftime("%Y-%m-%d %H:%M")+' 共处理'+str(loadprofile_spacechange.shape[1])+'维特征数据')

def analyze_space_change(request):
    spacechange_metric_data = pandas.read_csv('spacechange_metric_' + time.strftime("%d_%m_%Y") + '.csv')
    spacechange_metric_data = spacechange_metric_data[spacechange_metric_data['DIFF_KB']!=0]
    spacechange_metric_data = spacechange_metric_data[spacechange_metric_data['DIFF_KB'] > 0]
    spacechange_metric_data = spacechange_metric_data[spacechange_metric_data['Average Active Sessions'] != 0]

    #print(spacechange_metric_data['begin_time'])

    capicity_change = spacechange_metric_data['DIFF_KB']
    spacechange_metric_data.drop(['DIFF_KB','begin_time'],axis=1,inplace=True)

    # 采用PCA进行主成分分析
    from sklearn.decomposition import PCA
    pca = PCA(n_components=3)
    new_features = pca.fit_transform(spacechange_metric_data)
    print(new_features)


    x = spacechange_metric_data
    x_train_lasso, x_test_lasso, y_train_lasso, y_test_lasso = train_test_split(x, capicity_change, train_size=0.7, random_state=1)
    ss = StandardScaler()
    x_train_lasso = ss.fit_transform(x_train_lasso)
    x_test_lasso = ss.transform(x_test_lasso)

    model = Lasso()
    alpha = np.logspace(-3,2,10)
    np.set_printoptions(suppress=True)
    # print('alpha:{}'.format(alpha))
    #print(x_train.T)
    lasso_model = GridSearchCV(model,param_grid={'alpha': alpha}, cv=5)
    lasso_model.fit(x_train_lasso,y_train_lasso)
    y_hat_lasso = lasso_model.predict(x_test_lasso)

    # 采用FA和K-Means处理之后的特征，取前20个
    # ['Current Logons Count', 'Cell Physical IO Interconnect Bytes', 'Total PGA Allocated',
    # 'Physical Read Bytes Per Sec', 'Total PGA Used by SQL Workareas', 'Temp Space Used',
    # 'Physical Write Bytes Per Sec', 'Physical Write Total Bytes Per Sec', 'Cursor Cache Hit Ratio',
    # 'Redo Generated Per Sec', 'Redo Generated Per Txn', 'Logical Reads Per Sec', 'Rows Per Sort',
    # 'Physical Read Total Bytes Per Sec', 'Consistent Read Gets Per Txn', 'Network Traffic Volume Per Sec',
    # 'Physical Reads Direct Per Sec', 'DB Block Changes Per Sec', 'Logical Reads Per User Call',
    # 'Response Time Per Txn']

    #前30个
    # ['Total Sorts Per User Call', 'Cell Physical IO Interconnect Bytes', 'Total PGA Allocated',
    #  'Physical Read Bytes Per Sec', 'Temp Space Used', 'Total PGA Used by SQL Workareas',
    #  'Physical Write Bytes Per Sec', 'Physical Write Total Bytes Per Sec', 'Cursor Cache Hit Ratio',
    #  'Redo Generated Per Sec', 'Redo Generated Per Txn', 'Consistent Read Gets Per Sec', 'Rows Per Sort',
    #  'Physical Read Total Bytes Per Sec', 'Logical Reads Per Txn', 'Network Traffic Volume Per Sec',
    #  'Physical Reads Direct Per Sec', 'Logical Reads Per User Call', 'DB Block Changes Per Sec',
    #  'Logical Reads Per Sec', 'Database Time Per Sec', 'Physical Reads Per Sec', 'Unnamed: 0',
    #  'Physical Read Total IO Requests Per Sec', 'DB Block Changes Per Txn', 'Open Cursors Per Sec',
    #  'Consistent Read Gets Per Txn', 'Response Time Per Txn', 'Physical Reads Per Txn', 'Host CPU Utilization (%)']

    #35个特征
    # ['User Rollbacks Percentage', 'Cell Physical IO Interconnect Bytes', 'Total PGA Allocated',
    #  'Physical Read Total Bytes Per Sec', 'Temp Space Used', 'Total PGA Used by SQL Workareas',
    #  'Physical Write Total Bytes Per Sec', 'Physical Write Bytes Per Sec', 'Cursor Cache Hit Ratio',
    #  'Redo Generated Per Sec', 'Redo Generated Per Txn', 'Consistent Read Gets Per Sec', 'Rows Per Sort',
    #  'Logical Reads Per Txn', 'Physical Read Bytes Per Sec', 'Network Traffic Volume Per Sec',
    #  'Physical Reads Direct Per Sec', 'Logical Reads Per User Call', 'DB Block Gets Per Sec',
    #  'Logical Reads Per Sec', 'DB Block Changes Per Txn', 'Physical Reads Per Sec', 'Response Time Per Txn',
    #  'Physical Read Total IO Requests Per Sec', 'Unnamed: 0', 'Open Cursors Per Sec', 'Database Time Per Sec',
    #  'Physical Reads Per Txn', 'Consistent Read Gets Per Txn', 'Host CPU Utilization (%)',
    #  'Enqueue Requests Per Sec', 'DB Block Changes Per Sec', 'Total Index Scans Per Txn',
    #  'Executions Per User Call', 'Physical Writes Per Sec']

    #60个
    # ['Active Serial Sessions', 'Cell Physical IO Interconnect Bytes', 'Total PGA Allocated',
    #  'Physical Read Total Bytes Per Sec', 'Total PGA Used by SQL Workareas', 'Temp Space Used',
    #  'Physical Write Total Bytes Per Sec', 'Physical Write Bytes Per Sec', 'Cursor Cache Hit Ratio',
    #  'Redo Generated Per Sec', 'Redo Generated Per Txn', 'Consistent Read Gets Per Sec', 'Rows Per Sort',
    #  'Physical Read Bytes Per Sec', 'Consistent Read Gets Per Txn', 'Network Traffic Volume Per Sec',
    #  'Physical Reads Per Sec', 'DB Block Gets Per Sec', 'Logical Reads Per User Call', 'Physical Reads Direct Per Sec',
    #  'DB Block Gets Per Txn', 'I/O Requests per Second', 'Logical Reads Per Sec', 'Response Time Per Txn',
    #  'Open Cursors Per Sec', 'Unnamed: 0', 'Database Time Per Sec', 'Physical Reads Per Txn', 'Logical Reads Per Txn',
    #  'Host CPU Utilization (%)', 'Recursive Calls Per Sec', 'Txns Per Logon', 'Executions Per Txn',
    #  'Physical Writes Per Sec', 'Physical Reads Direct Per Txn', 'Total Index Scans Per Sec',
    #  'Total Index Scans Per Txn', 'DB Block Gets Per User Call', 'Physical Read IO Requests Per Sec',
    #  'Enqueue Requests Per Sec', 'DB Block Changes Per Sec', 'Full Index Scans Per Txn', 'Current Open Cursors Count',
    #  'Total Table Scans Per Txn', 'Database Wait Time Ratio', 'DB Block Changes Per Txn', 'User Calls Ratio',
    #  'User Commits Per Sec']

    #40个特征

    fa_k_spacechange_metric_data = spacechange_metric_data[
        ['Current Logons Count', 'Cell Physical IO Interconnect Bytes', 'Total PGA Allocated',
        'Physical Read Bytes Per Sec', 'Total PGA Used by SQL Workareas', 'Temp Space Used',
        'Physical Write Bytes Per Sec', 'Physical Write Total Bytes Per Sec', 'Cursor Cache Hit Ratio',
        'Redo Generated Per Sec', 'Redo Generated Per Txn', 'Logical Reads Per Sec', 'Rows Per Sort',
        'Physical Read Total Bytes Per Sec', 'Consistent Read Gets Per Txn', 'Network Traffic Volume Per Sec',
        'Physical Reads Direct Per Sec', 'DB Block Changes Per Sec', 'Logical Reads Per User Call',
        'Response Time Per Txn']]

    x = fa_k_spacechange_metric_data
    x_train, x_test, y_train, y_test = train_test_split(x, capicity_change, train_size=0.7, random_state=1)
    ss = MinMaxScaler()
    x_train = ss.fit_transform(x_train)
    x_test = ss.fit_transform(x_test)

    model = Ridge()
    alpha = np.logspace(-3, 2, 10)
    np.set_printoptions(suppress=True)
    # print('alpha:{}'.format(alpha))
    # print(x_train.T)
    ridge_model = GridSearchCV(model, param_grid={'alpha': alpha}, cv=5)
    ridge_model.fit(x_train, y_train)
    y_hat_ridge = ridge_model.predict(x_test)

    # final_model = pk.dumps(lasso_model)
    # f = open('lasso.txt','wb')
    # f.write(final_model)
    # f.close()
    # #print(x_train)
    # print('超参数：\n', lasso_model.best_params_)



    print(lasso_model.score(x_test_lasso, y_test_lasso))
    mse = np.average((y_hat_lasso - np.array(y_test_lasso)) ** 2)  # Mean Squared Error
    rmse = np.sqrt(mse)  # Root Mean Squared Error
    print(mse, rmse)



    print(ridge_model.score(x_test, y_test))
    mse = np.average((y_hat_ridge - np.array(y_test)) ** 2)  # Mean Squared Error
    rmse = np.sqrt(mse)  # Root Mean Squared Error
    print(mse, rmse)


    x_ind = np.arange(len(x_test))

    #print('y_test:{}'.format(y_test.shape()))
    #print('y_hat: {}'.format(y_hat.shape()))
    # import matplotlib as mpl
    # t = np.arange(len(x_test))
    # mpl.rcParams['font.sans-serif'] = [u'simHei']
    # mpl.rcParams['axes.unicode_minus'] = False
    # plt.figure(facecolor='w')
    # plt.plot(t, y_test, 'r-', linewidth=2, label=u'真实数据')
    # plt.plot(t, y_hat, 'g-', linewidth=2, label=u'预测数据')
    # plt.title(u'线性回归预测', fontsize=18)
    # plt.legend(loc='upper left')
    # plt.grid(b=True, ls=':')
    # plt.show()

    template = loader.get_template('spacechange_trend.html')
    REMOTE_HOST = '/static/assets/js'


    line_lasso = Line("空间容量每小时变化量预测-LASSO")
    line_lasso.add("真实数据", x_ind,y_test,is_smooth=True)
    line_lasso.add("预测数据", x_ind,y_hat_lasso,is_smooth=True)

    line_ridge = Line("空间容量每小时变化量预测-RIDGE")
    line_ridge.add("真实数据", x_ind, y_test, is_smooth=True)
    line_ridge.add("预测数据", x_ind, y_hat_ridge, is_smooth=True)

    context = dict(
        y_predict=y_hat_lasso,
        y_test = y_test,
        trend_line_lasso=line_lasso.render_embed(),
        trend_line_ridge=line_ridge.render_embed(),
        host=REMOTE_HOST,
        script_list=line_lasso.get_js_dependencies()
    )
    return HttpResponse(template.render(context, request))


#ARIMA 分析
from statsmodels.tsa.stattools import adfuller
from statsmodels.stats.diagnostic import  acorr_ljungbox
from statsmodels.tsa.stattools import acf,pacf
from statsmodels.tsa.arima_model import  ARIMA
from statsmodels.tsa.stattools import adfuller as ADF
def arima_spacechange_trend(request):
    spacechange_metric_data = pandas.read_csv('spacechange_metric_' + time.strftime("%d_%m_%Y") + '.csv',index_col=['begin_time'],header=0,parse_dates=['begin_time'])
    spacechange_metric_data = spacechange_metric_data[spacechange_metric_data['DIFF_KB'] != 0]


    load_result_columns = ['DIFF_KB', 'begin_time']
    load_result_all = pd.DataFrame(spacechange_metric_data, columns=load_result_columns)
    # df_sort = load_result.sort_values(['begin_time'])
    # df_sort.set_index(['begin_time'], inplace=True)
    # print(load_result['DIFF_KB'])
    # ADF平稳性 为了确定原始数据序列中没有随机趋势或确定趋势，需要对数据进行平稳性检验，否则将会产生“伪回归”的现象。本节采用ADF方法来进行平稳性检验。

    ###########################第一种ARIMA方法#######################
    # # 不使用最后5个数据
    # load_result = load_result_all.iloc[:len(load_result_all) - 30]
    # diff = 0
    # adf = ADF(load_result['DIFF_KB'])
    # # print(adf)
    # # adf[1]为p值，p值小于0.05认为是平稳的
    # while adf[1] >= 0.05:
    #     diff = diff + 1
    #     adf = ADF(load_result['DIFF_KB'].diff(diff).dropna())
    #     # print(adf)
    # print(u'原始序列经过%s阶差分后归于平稳，p值为%s' % (diff, adf[1]))
    #
    #
    #
    # #白噪声检验
    # # 为了验证序列中有用的信息是否已被提取完毕，需要对序列进行白噪声检验。如果序列检验为白噪声序列，
    # # 就说明序列中有用的信息已经被提取完毕了，剩下的全是随机扰动，无法进行预测和使用。本实验采用LB统计量的方法进行白噪声检验
    # from statsmodels.stats.diagnostic import acorr_ljungbox
    #
    # [[lb], [p]] = acorr_ljungbox(load_result['DIFF_KB'], lags=1)
    # if p < 0.05:
    #     print(u'原始序列为非白噪声序列，对应的p值为：%s' % p)
    # else:
    #     print(u'原始序列为白噪声序列，对应的p值为：%s' % p)
    #
    # [[lb], [p]] = acorr_ljungbox(load_result['DIFF_KB'].diff(1).dropna(), lags=1)
    # if p < 0.05:
    #     print(u'一阶差分序列为非白噪声序列，对应的p值为：%s' % p)
    # else:
    #     print(u'一阶差分为白噪声序列，对应的p值为：%s' % p)
    #
    # #模型识别
    # xdata = load_result['DIFF_KB']
    # from statsmodels.tsa.arima_model import ARIMA
    #
    # pmax = int(len(xdata) / 10)  # 一般阶数不超过length/10
    # qmax = int(len(xdata) / 10)
    # bic_matrix = []  # bic矩阵
    # for p in range(pmax + 1):
    #     tmp = []
    #     for q in range(qmax + 1):
    #         try:
    #             tmp.append(ARIMA(xdata, (p, 1, q)).fit().bic)
    #         except:
    #             tmp.append(None)
    #     bic_matrix.append(tmp)
    # bic_matrix = pd.DataFrame(bic_matrix)  # 取值区域
    # # stack()将数据from columns to indexs
    # p, q = bic_matrix.stack().astype('float64').idxmin()
    # print(u'p is q is:%s、%s' % (p, q))
    #
    #
    # #模型检验
    # lagnum = 12
    # xdata = load_result['DIFF_KB']
    # from statsmodels.tsa.arima_model import ARIMA
    #
    # arima = ARIMA(xdata, (p,1,q)).fit()
    # xdata_pred = arima.predict(typ='levels')  # predict
    # # print(xdata_pred)
    # pred_error = (xdata_pred - xdata).dropna()  # 残差
    #
    # from statsmodels.stats.diagnostic import acorr_ljungbox
    #
    # lb, p_l = acorr_ljungbox(pred_error, lags=lagnum)
    # h = (p_l < 0.05).sum()  # p值小于0.05，认为是非白噪声
    # # if h > 0:
    # #     print(u'模型ARIMA（%s,1,%s）不符合白噪声检验' % (p, q))
    # # else:
    # #     print(u'模型ARIMA（%s,1,%s）符合白噪声检验' % (p, q))
    #
    # #模型预测
    # test_predict = arima.forecast(30)[0]
    # # 预测对比
    # test_data = pd.DataFrame(columns=[u'实际容量', u'预测容量'])
    # test_data[u'实际容量'] = load_result_all[(len(load_result_all) - 30):]['DIFF_KB']
    # test_data[u'预测容量'] = test_predict
    # test_data = test_data.applymap(lambda x: '%.2f' % x)
    # print(test_data)
    #
    # #模型评价
    # # 计算误差
    # # 列操作
    # test_data[u'预测容量'] = test_data[u'预测容量'].astype(float)
    # test_data[u'实际容量'] = test_data[u'实际容量'].astype(float)
    # # 10**6单位换算
    # abs_ = (test_data[u'预测容量'] - test_data[u'实际容量']).abs() / 10 ** 6
    # mae_ = abs_.mean()
    # rmse_ = ((abs_ ** 2).mean()) ** 0.05
    # mape_ = (abs_ / test_data[u'实际容量']).mean()
    #
    # print(u'平均绝对误差为：%0.4f,\n均方根误差为：%0.4f,\n平均绝对百分误差为：%0.6f。' % (mae_, rmse_, mape_))


    ################第二种ARIMA方法##################################

    x = load_result_all['DIFF_KB'].astype(np.float)
    # x = np.log(x)
    #print(x)

    #x.drop([0])


    p = 0
    q = 1
    d = 1
    model = ARIMA(endog=x, order=(p, d, q))  # 自回归函数p,差分d,移动平均数q
    arima = model.fit(disp=-1)  # disp<0:不输出过程
    prediction = arima.fittedvalues

    test_predict = prediction.cumsum()
    mse = ((x - test_predict) ** 2).mean()
    rmse = np.sqrt(mse)
    print('rmse is %.4f',rmse)

    print(len(test_predict))
    print(len(x))



    line = Line("空间容量每小时变化量预测-ARIMA")
    template = loader.get_template('spacechange_trend.html')
    REMOTE_HOST = '/static/assets/js'
    # x_ind = np.arange(len(load_result_all[(len(load_result_all) - 30):]['DIFF_KB']))
    # line.add("真实数据", x_ind, load_result_all[(len(load_result_all) - 30):]['DIFF_KB'])

    x_ind = np.arange(len(x))
    predict_ind = np.arange(len(prediction))
    line.add("真实数据", x_ind, x)
    line.add("预测数据", predict_ind, prediction)
    # context = dict(
    #     y_predict=load_result_all[(len(load_result_all) - 30):]['DIFF_KB'],
    #     y_test=test_predict,
    #     trend_line_arima=line.render_embed(),
    #     host=REMOTE_HOST,
    #     script_list=line.get_js_dependencies()
    # )

    context = dict(
        y_predict=test_predict,
        y_test=x,
        trend_line_arima=line.render_embed(),
        host=REMOTE_HOST,
        script_list=line.get_js_dependencies()
    )

    return HttpResponse(template.render(context, request))



from analyze.Factor_Analyze import FactorAnalysis
from analyze.cluster import KMeansClusters, create_kselection_model
from analyze.preprocessing import (Bin, get_shuffle_indices,
                                    DummyEncoder,
                                    consolidate_columnlabels)
def Fa_kmeans(request):
    spacechange_metric_data = pandas.read_csv('spacechange_metric_' + time.strftime("%d_%m_%Y") + '.csv')
    spacechange_metric_data = spacechange_metric_data[spacechange_metric_data['DIFF_KB']!=0]
    spacechange_metric_data = spacechange_metric_data[spacechange_metric_data['Average Active Sessions'] != 0]

    #print(spacechange_metric_data['begin_time'])

    capicity_change = spacechange_metric_data['DIFF_KB']
    spacechange_metric_data.drop(['DIFF_KB','begin_time'],axis=1,inplace=True)

    n_rows, n_cols = spacechange_metric_data.shape

    #spacechange_metric_data =np.hstack(spacechange_metric_data)

    # # Bin each column (metric) in the matrix by its decile
    # binner = Bin(bin_start=1, axis=0)
    # binned_matrix = binner.fit_transform(spacechange_metric_data)
    #
    # # Shuffle the matrix rows
    # shuffle_indices = get_shuffle_indices(n_rows)
    # shuffled_matrix = binned_matrix[shuffle_indices, :]

    # Fit factor analysis model
    fa_model = FactorAnalysis()
    nonconst_columnlabels= spacechange_metric_data.columns.values.tolist()
    print(nonconst_columnlabels)
    # For now we use 5 latent variables
    fa_model.fit(spacechange_metric_data, nonconst_columnlabels, n_components=5)
    components = fa_model.components_.T.copy()
    print(fa_model.components_)

    # Run Kmeans for # clusters k in range(1, num_nonduplicate_metrics - 1)
    # K should be much smaller than n_cols in detK, For now max_cluster <= 20

    kmeans_models = KMeansClusters()
    kmeans_models.fit(components, min_cluster=1,
                      max_cluster=min(n_cols - 1, 35),
                      sample_labels=nonconst_columnlabels,
                      estimator_params={'n_init': 50})

    # Compute optimal # clusters, k, using gap statistics
    gapk = create_kselection_model("gap-statistic")
    gapk.fit(components, kmeans_models.cluster_map_)

    # Get pruned metrics, cloest samples of each cluster center
    pruned_metrics = kmeans_models.cluster_map_[gapk.optimal_num_clusters_].get_closest_samples()
    print(pruned_metrics)

    # Components: metrics * factors
    #components = fa_model.components_.T.copy()
    return HttpResponse(request)

#['Current Logons Count', 'Cell Physical IO Interconnect Bytes', 'Total PGA Allocated',
# 'Physical Read Bytes Per Sec', 'Total PGA Used by SQL Workareas', 'Temp Space Used',
# 'Physical Write Bytes Per Sec', 'Physical Write Total Bytes Per Sec', 'Cursor Cache Hit Ratio',
# 'Redo Generated Per Sec', 'Redo Generated Per Txn', 'Logical Reads Per Sec', 'Rows Per Sort',
# 'Physical Read Total Bytes Per Sec', 'Consistent Read Gets Per Txn', 'Network Traffic Volume Per Sec',
# 'Physical Reads Direct Per Sec', 'DB Block Changes Per Sec', 'Logical Reads Per User Call',
# 'Response Time Per Txn']












