#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2019/7/23
# @Author  : Edwin
# @Version : Python 3.6
# @File    : JSONUtile.py

from collections import OrderedDict
import json
class JSONUtil(object):

    @staticmethod
    def loads(config_str):
        return json.loads(config_str,
                          encoding="UTF-8",
                          object_pairs_hook=OrderedDict)

    @staticmethod
    def dumps(config, pprint=False, sort=False):
        indent = 4 if pprint is True else None
        if sort is True:
            if isinstance(config, dict):
                config = OrderedDict(sorted(config.items()))
            else:
                config = sorted(config)

        return json.dumps(config,
                          ensure_ascii=False,
                          indent=indent)
    #
    @staticmethod
    def dump(result,filename):
        return json.dump(result,filename)