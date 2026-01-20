Unicode1：

(a) 空串 `""`；

(b) `__str__(chr(0))` 给出空串，`__repr__(chr(0))` 给出 `'\x00'`，即用引号包裹的 unicode；

(c) 

```py
>>> chr(0)
'\x00'
>>> print(chr(0))

>>> "this is a test" + chr(0) + "string"
'this is a test\x00string'
>>> print("this is a test" + chr(0) + "string")
this is a teststring
```



Unicode2：

(a) 平均信息密度高（编码字母只需要 $1$ 字节），用 00 填充的情况更少，符合互联网文本存储格式；

(b) `b'\xf0\x9f\xa4\x93'`，emoji 🤓 的 Unicode。UTF-8 是变长的，任何在 UTF-8 下编码超过一字节的都会在 `.decode` 开头字节的时候直接报错。

(c) `b'\xf0\x9f`，`0xf0` 说明接下来还会有 3 byte 来一起编码一个字符，但接下来只有 1 byte 了，解码时报错。



train_bpe_tinystories：

(a) 

```
Time taken: 88.10 seconds (0.0245 hours)
Peak memory usage: 2.31 GB
Vocabulary size: 10000
Number of merges: 9743
Longest token: b' accomplishment' (15 bytes)
As string: ' accomplishment'
```

(b) 在 valid-set 上，pretokenize 花费 0.74s，train 花费 3.67s；在 train-set 上，pretokenize 花费 74.03s， train 花费 13.57s。



train_bpe_expts_owt：

需要算力。



