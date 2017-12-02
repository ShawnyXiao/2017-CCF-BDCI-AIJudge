# coding=utf-8
import time


def log(stri):
    now = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    print(str(now) + ' ' + str(stri))
