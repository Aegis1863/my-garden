---
{"dg-publish":true,"permalink":"/reinforcement-learning/12-muti-agent-rl/","dgPassFrontmatter":true}
---

[19\_多智能体.ipynb](https://github.com/Aegis1863/ML_practice/blob/master/%E5%BC%BA%E5%8C%96%E5%AD%A6%E4%B9%A0%E7%AC%94%E8%AE%B0/19_%E5%A4%9A%E6%99%BA%E8%83%BD%E4%BD%93.ipynb)
# 1. 多智能体强化学习

## 1.1. 多智能体和类似任务的状态设计

其实可以发现很多类似任务设计中，如果涉及比较复杂的状态，比如是数组的组合，复合数组等，都是“暴力”拼接、拉直的，类似 CNN 中把最后提取的特征展平到 MLP 一样，对人类来说这样做很难学到任何东西，但是对神经网络来说是有效的。

