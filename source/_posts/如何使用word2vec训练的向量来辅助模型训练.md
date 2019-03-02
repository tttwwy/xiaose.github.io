---
title: 如何使用word2vec训练的向量来辅助模型训练
tags:
  - 原创
  - 深度学习
abbrlink: 18396
date: 2016-08-15 10:00:00
---
使用word2vec工具在外部大规模语料上训练得到的向量，可以有效的辅助深度学习模型训练，提高结果。但是实际使用的时候，有很多方式可供选择。
- 直接用word2vec向量初始化模型embedding,训练的时候允许embedding向量更新。
这个方法最为常用，但是遇到不在训练语料中的词，就不能借助外部word2vec向量了。
- word2vec向量，先连接全连接层（可以是多层），转化后的向量再作为模型的embedding,训练的时候，word2vec向量保持不变，允许全连接层的参数更新。
这个方法，哪怕遇到不在训练语料中的词，只要这个词在外部大规模语料中，能得到word2vec向量，那么就没问题。同时因为word2vec向量在训练的时候固定，因此模型训练涉及的参数会大大减少。
<!-- more -->
因为word2vec向量的分布，和模型实际需要的向量分布，可能存在差异，因此这个全连接层的作用，就是对word2vec向量的分布进行调整，让他尽可能接近模型需要的向量分布。
- 将word2vec向量拷贝，得到向量A和向量B，训练的时候，向量A保持不变，允许向量B的参数更新，最终embedding向量是A和B的平均。
这个idea的想法，其实是限制word2vec向量的调整，避免调整的时候，太偏离原始向量。
- 1. 第一轮训练，用Google News训练好的Word2vec初始化embedding,对未知词随机初始化embedding,在训练的时候,固定住word2vec初始化的embedding,而允许未知词的embedding进行调整
    2. 第二轮训练，允许所有embedding调整,继续训练
这个idea也可以很好的处理未知词，第一轮的时候，因为固定了word2vec向量，因此模型会尽可能基于word2vec向量的分布来调整自己的参数。但是可能分布差异太大，导致模型参数无论怎么调整，都得不到最好结果。因此第二轮的时候，允许word2vec向量进行适当调整。

- 参考论文
    1. 2016-AAAI-Building End-to-End Dialogue Systems Using Generative Hierarchical Neural Network Models
    2. 2014-EMNLP-Convolutional Neural Networks for Sentence Classification