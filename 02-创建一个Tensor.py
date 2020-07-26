import tensorflow as tf
import numpy as np

# 直接从 numpy 或 list 转换而来
# tf.convert_to_tensor() 与 tf.constant() 功能相似
print(tf.convert_to_tensor([1, 2, 3]))  # 类型为 int32
print(tf.convert_to_tensor([1, 2, 3.]))  # 自动全部升级到 float32
try:
    print(tf.convert_to_tensor([1, 2, "hello."]))  # Tensor 类型不统一, 无法转换
except Exception as e:
    print(e)
try:
    print(tf.convert_to_tensor([[1, 2], [2]]))  # 元素个数不统一, 无法转换
except Exception as e:
    print(e)

print(tf.convert_to_tensor(np.ones([3, 4, 2])))

print("----------")

# tf.zeros() / tf.ones() / tf.fill()

print(tf.zeros([]))  # 注意此处传入的是 shape
print(tf.zeros([3]))
print(tf.zeros([2, 3]))
print(tf.zeros([2, 3, 3]))

print(tf.zeros_like([1, 2, 3]))  # 此处传入的是任意一个 Tensor, 会返回一个与该 Tensor 的 shape 相同的填充满 0 的 Tensor.

print(tf.ones([3]))  # 以下与 tf.zeros() 同理
print(tf.ones_like([1, 2]))

print(tf.fill([2, 2], 0))  # 等价于 tf.zeros([2, 2])
print(tf.fill([3, 3], 9))

print("----------")

# tf.random.normal() 正态分布
print(tf.random.normal([3, 3], mean=1, stddev=1))  # mean 均值, stddev 标准差
print(tf.random.normal([3, 3]))  # 默认 mean = 0, stddev = 1

# 使用截断的正态分布防止梯度消失
print(tf.random.truncated_normal([3, 3]))  # 大于两个标准差的数据被舍去

print("----------")

# tf.random.unifrom() 均匀分布
print(tf.random.uniform([2, 2], minval=0, maxval=1))  # 从 [0, 1) 的均匀分布
print(tf.random.uniform([2, 2], minval=0, maxval=100, dtype=tf.int32))  # 从 [0, 100) 的均匀分布
print(tf.random.uniform([2, 2]))  # 默认为 [0, 1) 的均匀分布

print("----------")

# 小应用: 随机打散
idx = tf.range(10)
idx = tf.random.shuffle(idx)
print(idx)

a = tf.random.uniform([10, 2], maxval=10, dtype=tf.int32)
print(a)
print(tf.gather(a, idx))
