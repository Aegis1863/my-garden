---
{"dg-publish":true,"permalink":"/reinforcement-learning/4-advantage-actor-critic-a2-c/","dgPassFrontmatter":true,"created":"2023-08-07T17:26:17.787+08:00"}
---


代码 [10\_Actor-Critic.ipynb](https://github.com/Aegis1863/ML_practice/blob/master/%E5%BC%BA%E5%8C%96%E5%AD%A6%E4%B9%A0%E7%AC%94%E8%AE%B0/10_Actor-Critic.ipynb)

#机器学习/强化学习/同策略 #机器学习/强化学习/连续动作 #机器学习/强化学习/离散动作 

# 1. 演员评论员框架

评论员网络，也就是价值网络 V，采用和 [[Reinforcement Learning/1_DQN\|1_DQN]] 一样的时序差分的方式更新参数，即估计的本轮 Q 值往(本轮奖励+估计的下一轮 Q 值)的方向更新。

演员网络，即策略网络，用 [[Reinforcement Learning/2_Policy Gradient\|2_Policy Gradient]] 的方式更新。

可以用于连续动作，但不建议，可能需要极大的训练量，原因可能是states序列相关造成critic难以收敛；**建议用于离散动作**。

# 2. 优势函数推导
{ #8889c9}

[[Reinforcement Learning/1_DQN#^b4e4d9\|优势函数]] 的思想在 Dueling DQN 中已经提到过了，那里需要 Q 值，所以把 $A = Q - V​​$ 变成 $Q=V+A$，其中 A 也是由 Q 网络的一部分估计的，这里是改进的优势函数，需要的就是 $A = Q - V​​$，其中 Q 来自演员网络，V 来自评论员网络。

目标函数就是动作的概率乘以优势函数 A，即 $log\_prob(action)* A$，**推导待补充**，优势函数原本是每轮以及后续获得的折算奖励（G）减去一个基线 b，也就是 $A = G-b$，但是这个折算是比较麻烦的，因为有 $E(G) = Q_{\pi_{\theta}}(s^n_t,a^n_t)$，所以可以考虑用 Q 替代 G，而 $b$ 一般是状态价值函数 $V_{\pi_{\theta}}(s_t)$，因此优势函数变成
$$A = Q_{\pi_{\theta}}(s^n_t,a^n_t)-V_{\pi_{\theta}}(s_t)$$
可以用 V 来估计 Q，即有 $Q_{\pi}(s^n_t,a^n_t)=E(r_t^n+V_{\pi_{\theta}}(s_{t+1}))$ ，把这个式子带入上式，并且这里先去掉期望，所以有
$$A^{\theta}\left(s_t,a_t\right) = r_t^n+V_{\pi}(s_{t+1}^n) - V_{\pi}(s_t)$$
期望值很难算，经过多次实验，去掉期望值容易计算，并且效果还可以，因此在这里一般就去掉期望，带入最开始给的优势函数。

这样是很容易代码实现的。在代码里面，优势 G 就是 `td_delta`，再乘以负的动作概率的对数，就等于演员网络的损失，对其参数求梯度，就实现梯度上升。 #机器学习/强化学习/优势函数推导 
{ #9e8ce5}
