# Generated by Django 2.1 on 2019-01-15 06:53

from django.db import migrations


class Migration(migrations.Migration):

    dependencies = [
        ('ora_dual', '0006_userinfo_user_group'),
    ]

    operations = [
        migrations.RenameField(
            model_name='userinfo',
            old_name='user_group',
            new_name='usergroup',
        ),
    ]
