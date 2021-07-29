import os
path=os.path.abspath(os.path.dirname(__file__))
print('***获取当前目录***')
print(path)
os.mkdir(path+"./now")
print('***获取上上级目录***')
path_up=os.path.abspath(os.path.join(path, "../.."))
os.mkdir(path_up+"./up")
