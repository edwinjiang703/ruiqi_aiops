# Generated by Django 2.1 on 2019-01-23 09:05

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('ora_dual', '0011_spaceusage_collect_time'),
    ]

    operations = [
        migrations.AlterField(
            model_name='spaceusage',
            name='collect_time',
            field=models.CharField(max_length=20),
        ),
    ]