# -*- coding: utf-8 -*-
# @Time    : 2019/1/20 8:03 AM
# @Author  : Edwin
# @File    : display_charts.py
# @Software: PyCharm

#pyecharts with django
import math
from django.http import HttpResponse
from django.template import loader
from pyecharts import Line3D,Bar,Timeline,Pie
from ora_dual import models

# from pyecharts.constants import DEFAULT_HOST #这句去掉
REMOTE_HOST = '/static/assets/js'


def line3d_echart(request):
    template = loader.get_template('pyecharts.html')
    l3d = line3d()
    context = dict(
        myechart=l3d.render_embed(),
        # host=DEFAULT_HOST,#这句改为下面这句
        host=REMOTE_HOST,  # <-----修改为这个
        script_list=l3d.get_js_dependencies()
    )
    return HttpResponse(template.render(context, request))


def line3d():
    _data = []
    for t in range(0, 25000):
        _t = t / 1000
        x = (1 + 0.25 * math.cos(75 * _t)) * math.cos(_t)
        y = (1 + 0.25 * math.cos(75 * _t)) * math.sin(_t)
        z = _t + 2.0 * math.sin(75 * _t)
        _data.append([x, y, z])
    range_color = [
        '#313695', '#4575b4', '#74add1', '#abd9e9', '#e0f3f8', '#ffffbf',
        '#fee090', '#fdae61', '#f46d43', '#d73027', '#a50026']
    line3d = Line3D("3D line plot demo", width=1200, height=600)
    line3d.add("", _data, is_visualmap=True,
               visual_range_color=range_color, visual_range=[0, 30],
               is_grid3D_rotate=True, grid3D_rotate_speed=180)
    return line3d
#
# from pyecharts import Bar
# def load_profile_trend(request):
#     bar =Bar("我的第一个图表", "这里是副标题")
#     bar.add("服装", ["衬衫", "羊毛衫", "雪纺衫", "裤子", "高跟鞋", "袜子"], [5, 20, 36, 10, 75, 90])
#     bar.show_config()
#     bar.render()
#     return HttpResponse(bar)

from datetime import datetime
def bar_echart(request,*args,**kwargs):
    template = loader.get_template('./node_modules/gentelella/production/display_metric_detail.html')
    snap = request.GET.get('snapdate')
    #snap_date = datetime.strptime(snap, '%y/%m/%d').strftime('%Y-%m-%d')
    if snap:

        load_profile_per_hour = list(models.loadmetric_hour.objects.values("time","redo_second", "logical_second", "physical_second", "execs_second", "trans_second").filter(snap_date=snap).all())
        space_usage = list(models.spaceusage.objects.values("tablespace_name","percent").filter(collect_time=snap).all())
        print(space_usage)

        # load_profile_obj = apps.get_model('ora_dual', 'loadmetric_hour')
        # load_profile_field = load_profile_obj._meta.fields
        # title = []
        # for ind in range(len(load_profile_field)):
        #     title.append(load_profile_field[ind].name)
        attr = []

        for key,value in load_profile_per_hour[0].items():
            attr.append(key)

        val_usage = []
        val_name = []
        for idx in range(len(space_usage)):
            val_name.append(space_usage[idx]['tablespace_name'])
            val_usage.append(space_usage[idx]['percent'])

        usage_pie = Pie("饼图-空间使用率", title_pos='center')
        usage_pie.add(
            "",
            val_name,
            val_usage,
            radius=[40, 75],
            label_text_color=None,
            is_label_show=True,
            legend_orient="vertical",
            legend_pos="left",
        )
        # pie.render()
        timeline = Timeline(is_auto_play=True, timeline_bottom=0)

        for idx in range(len(load_profile_per_hour)):
            val = []
            for key,value in load_profile_per_hour[idx].items():
                val.append(value)

            bar = Bar("数据库指标", val[0])
            bar.add("值/秒", attr[1:], val[1:])
            timeline.add(bar, val[0])

        context = dict(
            snap_date = snap,
            title = attr,
            usage_pie = usage_pie.render_embed(),
            space_usage = space_usage,
            metric_data = load_profile_per_hour,
            myechart=timeline.render_embed(),
            # host=DEFAULT_HOST,#这句改为下面这句
            host=REMOTE_HOST,  # <-----修改为这个
            script_list=timeline.get_js_dependencies()
        )
        return HttpResponse(template.render(context, request))



