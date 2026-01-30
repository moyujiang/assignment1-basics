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

(a)

```
Time taken: 12932.71 seconds (3.5924 hours)
Pretokenize time: 525.41 seconds
Train time: 8944.50 seconds
Peak memory usage: 28.58 GB
Vocabulary size: 32000
Number of merges: 31743

Longest token: b'\xc3\x83\xc3\x82\xc3\x83\xc3\x82\xc3\x83\xc3\x82\xc3\x83\xc3\x82\xc3\x83\xc3\x82\xc3\x83\xc3\x82\xc3\x83\xc3\x82\xc3\x83\xc3\x82\xc3\x83\xc3\x82\xc3\x83\xc3\x82\xc3\x83\xc3\x82\xc3\x83\xc3\x82\xc3\x83\xc3\x82\xc3\x83\xc3\x82\xc3\x83\xc3\x82\xc3\x83\xc3\x82' (64 bytes)
As string: 'ÃÂÃÂÃÂÃÂÃÂÃÂÃÂÃÂÃÂÃÂÃÂÃÂÃÂÃÂÃÂÃÂ'
```

(b)

网络数据更难清洗，导致 owt 上的最长 token 不是有意义的单词；owt 数据集显著大于 TinyStories，这导致 merge 阶段慢很多。



tokenizer_experiments：

(a)

TinyStories tokenizer (10K) compression ratio (bytes/token): 4.0908
OpenWebText tokenizer (32K) compression ratio (bytes/token): 4.5357

(b)

OpenWebText sample with TinyStories tokenizer (bytes/token): 3.3438
OWT token count multiplier (TinyStories vs OWT tokenizer): 1.356x
OWT bytes/token change vs OWT tokenizer: -26.28%

TinyStories 中多简短故事，用词相对简单，且 vocab 更小，对 owt 中的少见词汇、URL 模式等不能很好地匹配，被切为更多的 tokens，压缩率显著下降。

(c)

OWT tokenizer 在 OWT 上约 ~1.5 MB/s，用这个吞吐量粗估 Pile 825GB 时间：$t \approx \frac{825\times 10^9}{1.5\times 10^6}s\approx 6d$，约 $6$ 天。

(d)

