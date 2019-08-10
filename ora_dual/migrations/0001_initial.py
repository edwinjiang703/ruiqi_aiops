# -*- coding: utf-8 -*-
# Generated by Django 1.10.1 on 2019-01-10 10:01
from __future__ import unicode_literals

from django.db import migrations, models


class Migration(migrations.Migration):

    initial = True

    dependencies = [
    ]

    operations = [
        migrations.CreateModel(
            name='io_capacity_predict',
            fields=[
                ('id', models.AutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('predict_value', models.FloatField()),
                ('name', models.CharField(max_length=126)),
            ],
        ),
        migrations.CreateModel(
            name='io_data',
            fields=[
                ('id', models.AutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('period', models.CharField(max_length=16)),
                ('true_value', models.FloatField()),
                ('desc', models.CharField(max_length=126)),
            ],
        ),
    ]
