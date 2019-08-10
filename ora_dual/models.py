from django.db import models

# Create your models here.

class io_data(models.Model):
    period = models.CharField(max_length=16)
    true_value = models.FloatField()
    desc = models.CharField(max_length=126)

class io_capacity_predict(models.Model):
    predict_value = models.FloatField()
    name = models.CharField(max_length=126)

class UserInfo(models.Model):
    # db_index unique  unique_for_date   unique_for_month  unique_for_year
    # verbose_name Admin 中显示的字段名称
    # blank admin中是否允许用户输入为空
    # editable admin中是否可以编辑
    # help_text admin中该字段的提示信息
    # choices admin中显示选择框的内容
    # validators 自定义错误验证  可以定制想要的验证规则
    username = models.CharField(max_length=32,verbose_name='用户名',blank=True,help_text='username')
    age = models.IntegerField()
    gender = models.CharField(max_length=12)
    user_choice = (
        (1,'Super User'),
        (2,'VIP User'),
        (3,'Normal User')
    )
    user_type = models.CharField(max_length=32,choices=user_choice,default=3)
    usergroup = models.ForeignKey("UserGroup",to_field="id",on_delete=models.CASCADE)


class UserGroup(models.Model):
    #db_index unique  unique_for_date   unique_for_month  unique_for_year
    #verbose_name Admin 中显示的字段名称
    #blank admin中是否允许用户输入为空
    #editable admin中是否可以编辑
    #help_text admin中该字段elapse(min)的提示信息
    #choices admin中显示选择框的内容
    #validators 自定义错误验证  可以定制想要的验证规则

    id = models.AutoField(primary_key=True)
    caption = models.CharField(max_length=32,db_column='cp')

class loadmetric_hour(models.Model):
    instance_number = models.IntegerField()
    snap_date = models.DateField()
    time = models.CharField(max_length=10)
    elapse_min = models.FloatField()
    dbtime_min = models.FloatField()
    redo = models.FloatField()
    redo_second = models.FloatField()
    logical = models.FloatField()
    logical_second  = models.FloatField()
    physical = models.FloatField()
    physical_second = models.FloatField()
    execs = models.FloatField()
    execs_second = models.FloatField()
    parse = models.FloatField()
    parse_second = models.FloatField()
    hardware = models.FloatField()
    harware_second = models.FloatField()
    trans = models.FloatField()
    trans_second = models.FloatField()

class spaceusage(models.Model):
    collect_time = models.CharField(max_length=30)
    #data_time = models.DateField()
    tablespace_name = models.CharField(max_length=10)
    total = models.FloatField()
    free = models.FloatField()
    used = models.FloatField()
    percent = models.FloatField()

class system_metric_period(models.Model):
    collect_time = models.CharField(max_length=30)
    begin_time = models.CharField(max_length=30)
    end_time = models.CharField(max_length=30)
    metric_name = models.CharField(max_length=100)
    data_value = models.CharField(max_length=100,null=True)
    metric_average = models.FloatField(null=True)
    metric_standard = models.FloatField(null=True)
    metric_squares = models.FloatField(null=True)


class spacechange(models.Model):
    collect_time = models.CharField(max_length=30)
    # data_time = models.DateField()
    tablespace_name = models.CharField(max_length=10)
    tablespace_usedsize_kb = models.FloatField()
    tablespace_size_kb = models.FloatField()
    DIFF_KB = models.FloatField()


class mysql_system_metric_period(models.Model):
    collect_time = models.CharField(max_length=30)
    metric_name = models.CharField(max_length=100)
    data_value = models.CharField(max_length=100,null=True)




