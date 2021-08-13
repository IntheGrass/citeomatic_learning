# 阅读源码中遇到的知识点整理
## **函数**
### pad_sequences()
> 简述： 填充序列使文本集中所有文本长度相同。
原型:
```python
from keras.preprocessing.sequence import pad_sequences
pad_sequences(
    sequences, # 序列列表（2D）
    maxlen=None, # 序列最大长度
    dtype='int32', # 类型
    padding='pre', # pre||post，分别表示前||后填充
    truncating='pre', # pre||post，分别表示大于最大长度时从前||后截取
    value=0.0)
```
### setattr(), getattr()
> 简述：python内置函数, 分别为设置/获取对象属性值
```python
setattr(object, name, value)

getattr(object,name[, 
        default # default为默认返回值，如果无default且属性不存在则会出发AttributeError]
    )
```
## **工具包**
### resource
> 简述： 提供程序测量和控制系统资源的基本机制，用符号常数来指定特定的资源

**_unix only，在window下会报错:_** `ModuleNotFoundError: No module named 'resource'`
### arrow
> 简述： 日期时间处理库
### spacy
> 简述： NLP工具包
### whoosh
> 简述： 轻量级搜索库
### mmh3
> 简述：全称murmurhash 3 ,提供一种非加密的哈希算法
### tensorflow
