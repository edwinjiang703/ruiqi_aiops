# -*- coding: utf-8 -*-
# @Time    : 2019/1/5 1:15 PM
# @Author  : Edwin
# @File    : base.py
# @Software: PyCharm

from abc import ABCMeta, abstractmethod


class ModelBase(object, metaclass=ABCMeta):

    @abstractmethod
    def _reset(self):
        pass

