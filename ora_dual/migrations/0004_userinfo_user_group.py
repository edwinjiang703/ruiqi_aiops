# Generated by Django 2.1 on 2019-01-15 06:38

from django.db import migrations, models
import django.db.models.deletion


class Migration(migrations.Migration):

    dependencies = [
        ('ora_dual', '0003_auto_20190115_0618'),
    ]

    operations = [
        migrations.AddField(
            model_name='userinfo',
            name='user_group',
            field=models.ForeignKey(default=2, on_delete=django.db.models.deletion.CASCADE, to='ora_dual.UserGroup'),
            preserve_default=False,
        ),
    ]
