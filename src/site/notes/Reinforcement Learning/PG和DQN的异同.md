---
{"dg-publish":true,"permalink":"/reinforcement-learning/pg-dqn/","dgPassFrontmatter":true,"created":"2023-08-17T23:08:28.997+08:00","updated":"2023-08-21T12:52:07.700+08:00"}
---

{ #f11378}


[[Reinforcement Learning/DQN#^0cba58\|深度Q网络]]输出的是价值，所以有时候也说成价值网络V，它的目的是评价当前状态，采取各个离散动作的价值；或者，如果是连续动作下，它评价的是当前状态和采取某个动作的价值。
#机器学习/强化学习/Q网络 

[[Reinforcement Learning/PG\|策略网络]]输出的是动作，可以直接输出动作，连续动作可以用tanh映射输出并且缩放到动作空间上，也可以输出均值方差来构造一个连续或离散的分布，然后从里面抽动作；离散动作还可以直接通过softmax输出动作概率，再抽取动作。
#机器学习/强化学习/策略网络