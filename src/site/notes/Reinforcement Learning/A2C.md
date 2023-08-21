---
{"dg-publish":true,"permalink":"/reinforcement-learning/a2-c/","dgPassFrontmatter":true}
---


代码 [10\_Actor-Critic.ipynb](https://github.com/Aegis1863/ML_practice/blob/master/%E5%BC%BA%E5%8C%96%E5%AD%A6%E4%B9%A0%E7%AC%94%E8%AE%B0/10_Actor-Critic.ipynb)

#机器学习/强化学习/同策略 #机器学习/强化学习/连续动作 #机器学习/强化学习/离散动作 

# 演员评论员框架

评论员网络，也就是价值网络V，采用和[[Reinforcement Learning/DQN\|DQN]]一样的时序差分的方式更新参数，即估计的本轮Q值往(本轮奖励+估计的下一轮Q值)的方向更新。

演员网络，即策略网络，用[[Reinforcement Learning/PG\|PG]]的方式更新。

**可以用于连续动作，但不建议**，可能需要极大的训练量，原因可能是states序列相关造成critic难以收敛；**建议用于离散动作**。

# 优势函数推导
{ #8889c9}


目标函数就是动作的概率乘以优势函数A，优势函数原本是每轮以及后续获得的奖励（折算后的，称为G）减去一个基线b，也就是$A = G-b$，但是这个折算会麻烦一点，可以考虑用Q替代，因为有$E(G) = Q_{\pi_{\theta}}(s^n_t,a^n_t)$，而$b$一般是状态价值函数$V_{\pi_{\theta}}(s_t)$，因此优势函数变成

$$Q_{\pi_{\theta}}(s^n_t,a^n_t)-V_{\pi_{\theta}}(s_t)$$
同时有$E(Q_{\pi}(s^n_t,a^n_t))=V_{\pi_{\theta}}(s_t)$，又$Q$存在

$$Q_{\pi}(s^n_t,a^n_t) = r_t^n+Q_{\pi}(s_{t+1}^n,a_{t+1}^n)$$
因此（后面省去$\pi$的$\theta$角标）
$$Q_{\pi}(s^n_t,a^n_t) = r_t^n+Q_{\pi}(s_{t+1}^n,a_{t+1}^n) = E(r_t^n+V_{\pi}(s_{t+1}^n))$$
期望值很难算，经过多次实验，去掉期望值容易计算，并且效果，因此在这里一般就去掉期望，带入最开始给的优势函数，最终优势A就变成
$$A^{\theta}\left(s_t,a_t\right) = r_t^n+V_{\pi}(s_{t+1}^n) - V_{\pi_{\theta}}(s_t)$$
代码里面，优势G就是`td_delta`，再乘以负的动作概率的对数，就等于演员网络的损失，对其参数求梯度，就实现梯度上升。 #机器学习/强化学习/优势函数推导 
{ #9e8ce5}
