---
title: >-
  Character-based Joint Segmentation and POS Tagging for Chinese using
  Bidirectional RNN-CRF
tags:
  - 原创
  - 深度学习
  - NLP
abbrlink: 27292
date: 2017-04-06 10:00:00
---
# [Character-based Joint Segmentation and POS Tagging for Chinese using Bidirectional RNN-CRF](https://arxiv.org/pdf/1704.01314.pdf)
## 作者
Yan Shao and Christian Hardmeier and Jorg Tiedemann  and Joakim Nivre

## 关键词
Bi-RNN-CRF，分词，词性标注，汉字embedding<!-- more -->
## 文章来源
https://arxiv.org/pdf/1704.01314.pdf
## 问题
如何联合做中文分词和词性标注。
## 模型
![](/images/15515449304888.jpg)

这篇论文首先将双向GRU+CRF这种非常流行的序列标注模型应用到了分词和词性标注任务上。使用了联合训练的方式，即将分词的标签和词性标注的标签拼在一起，同时输出。例如**夏天太热**这个句子，分词的标签输出是B E S S，词性标注的标签输出是NT NT AD VA,那么联合的标签输出就是B-NT E-NT S-AD S-VA。

同时，这篇论文使用了三种汉字embedding方式
- Concatenated N-gram：即使用包含这个字的n-gram词，来作为这个字的embedding,这样相比双向GRU的方式，可以更直接的考虑局部信息
- Radicals and Orthographical Features：首先使用了部首信息，作为embedding，其次将汉字作为图像送给CNN，获取字的embedding
- 公开语料预训练好的字embedding

论文decoding阶段，使用了ensemble方法，把4个不同初始化方式的模型取平均，进行decoding

## 资源
代码：[https://github.com/yanshao9798/tagger](https://github.com/yanshao9798/tagger "https://github.com/yanshao9798/tagger")
## 简评
这篇文章将分词和词性标注进行联合，避免了两个任务间的错误传递问题。同时针对汉字特有的特点，探索了基于部首和汉字图像的embedding方式，对其他中文处理任务也有一定的启发。



