---
{"dg-publish":true,"permalink":"/reinforcement-learning/12-muti-agent-rl/","dgPassFrontmatter":true,"created":"2023-10-20T15:22:25.590+08:00","updated":"2023-10-21T19:14:07.064+08:00"}
---

[19\_多智能体.ipynb](https://github.com/Aegis1863/ML_practice/blob/master/%E5%BC%BA%E5%8C%96%E5%AD%A6%E4%B9%A0%E7%AC%94%E8%AE%B0/19_%E5%A4%9A%E6%99%BA%E8%83%BD%E4%BD%93.ipynb)
# 1. 多智能体强化学习

## 1.1. 多智能体和类似任务的状态设计

可以发现很多类似任务设计中，如果是比较复杂的状态，比如是数组的组合，复合数组等，都是“暴力”拼接、拉直的，类似 [[MLP-ABC/06_其他机器学习技术#卷积神经网络\|CNN]] 中把最后提取的特征展平到 [[MLP-ABC/02_神经网络\|MLP]] 一样，对人类来说这样做很难学到任何东西，但是对神经网络来说是有效的。

## 1.2. 多智能体编程技巧

其实很简单，比如算法是 [[Reinforcement Learning/5_PPO\|PPO]]，只需要建立一个列表，装入若干 [[Reinforcement Learning/5_PPO\|PPO]] 算法即可，可以采取中心化训练，去中心化执行的方法：评论员网络接收全部智能体状态（或和动作，全部拼接、展平），每个智能体，仅采用它观察到的状态采取动作。

因此，进一步应该想到，每个智能体的状态应该单独给出，整理在一个表里输出作为总状态。并且在训练智能体的时候，善用 `enumerate()` 和 `zip()`，同时输出序号和内容，便于和不同智能体数据对应。

要注意的是，由于输出是包含多组智能体输出的，所以比单智能体操作不一样：在经验池采样一个批量只会，需要先转置，再把倒数第二维改成 tensor，相当于变成多个单智能体的状态，装在一个大列表中。

下面是一个例子

```python
# 环境给出的状态形式，每行代表某一时刻三个智能体的观测，这里假设是随机抽了三个时点数据作为一个批量，这三个时点对应三行

x = [[np.array([1, 2]), np.array([3, 4]), np.array([3, 5])],
     [np.array([4, 2]), np.array([0, 7]), np.array([5, 5])],
     [np.array([1, 5]), np.array([4, 7]), np.array([6, 2])],]
```

转置为

```python
[[array([1, 2]), array([4, 2]), array([1, 5])], 
 [array([3, 4]), array([0, 7]), array([4, 7])], 
 [array([3, 5]), array([5, 5]), array([6, 2])]]
```

再修改为

```python
[tensor([[1., 2.], [4., 2.], [1., 5.]]), 
 tensor([[3., 4.], [0., 7.], [4., 7.]]), 
 tensor([[3., 5.], [5., 5.], [6., 2.]])]
```

要实现该操作，可以参考以下函数写法

```python
def stack_array(x):
	rearranged = [[sub_x[i] for sub_x in x]
				  for i in range(len(x[0]))]
	return [
		torch.FloatTensor(np.vstack(aa)).to(device)
		for aa in rearranged
	]
```

