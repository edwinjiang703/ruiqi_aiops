# Generated by Django 2.1 on 2019-01-23 09:08

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('ora_dual', '0012_auto_20190123_0905'),
    ]

    operations = [
        migrations.AlterField(
            model_name='spaceusage',
            name='collect_time',
            field=models.CharField(max_length=30),
        ),
    ]
