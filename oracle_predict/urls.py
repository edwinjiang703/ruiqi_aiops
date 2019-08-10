"""oracle_predict URL Configuration

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/1.10/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  url(r'^$', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  url(r'^$', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.conf.urls import url, include
    2. Add a URL to urlpatterns:  url(r'^blog/', include('blog.urls'))
"""
from django.conf.urls import url,re_path,include
from django.contrib import admin
from ora_dual import views
from django.urls import path


urlpatterns = [
    path('ora_dual/',include('ora_dual.urls')),
    path('metric_process/',include('metric_process.urls'))
    #url(r'^metric_process/',include('metric_process.urls'))
    # url(r'^admin/', admin.site.urls),
    # url(r'^index/',views.index),
    # url(r'^task_res/',views.task_res),
    # url(r'^login/',views.login),
    # url(r'^home_long_name_home/',views.home,name='ihome'),#name 是别名，可以在view里面替换。
    # #CBV Class Base View
    # url(r'^TestCBV/',views.TestCBV.as_view()),
    # #引入正则表达式
    # re_path('detail-(?P<uid>\d+)-(?P<id>\d+).html/',views.detail)

]
