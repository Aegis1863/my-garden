---
{"dg-publish":true,"permalink":"/reinforcement-learning/td-3/","dgPassFrontmatter":true}
---

代码 [13\_TD3.ipynb](https://github.com/Aegis1863/ML_practice/blob/master/%E5%BC%BA%E5%8C%96%E5%AD%A6%E4%B9%A0%E7%AC%94%E8%AE%B0/13_TD3.ipynb)

#机器学习/强化学习/连续动作 #机器学习/强化学习/异策略 #机器学习/强化学习/确定性策略

> twin delayed DDPG

1. 第一个技巧是，采用了两个评论员网络，谁给出的目标Q值小，就用谁的Q值，评论员网络也就是[[Reinforcement Learning/DQN\|Q网络]];

2. 第二个技巧是，目标网络需要延迟更新，在[[Reinforcement Learning/DDPG\|DDPG]]之前的[[Reinforcement Learning/DQN\|DQN]]中就已经这样操作了; 

3. 第三个技巧是，这里策略（演员）网络的更新也要延迟，策略网络更新太快会导致输出的动作很不稳定。并且更新方法采用和[[Reinforcement Learning/DDPG#^5661e0\|DDPG]]一样的软更新;

* 即时更新
    - 原评论员网络 `critic_1` 和 `critic_2` **直接梯度下降**
* 延迟更新
    - 原演员网络 `actor` **直接梯度下降**
    - 目标评论员网络 `target_critic_1` 和 `target_critic_2` **软更新**
    - 目标演员网络 `target_actor` **软更新**
