import tensorflow as tf
import numpy as np

# 测试一下 Tensor 的类型
t = tf.constant(1)
print(t)

t = tf.constant(1.)
print(t)

# 不匹配的类型会出错
try:
    t = tf.constant(2.2, dtype=tf.int32)
except Exception as e:
    print(e)

# double 是 float64 的别名
t = tf.constant(2., dtype=tf.double)
print(t)

# Tensorflow 支持 bool 型变量
t = tf.constant([True, False])
print(t)

# Tensorflow 支持 string 型变量
t = tf.constant("Helloworld!")
print(t)

print("----------")

# 测试 Tensor 的属性
t = tf.constant([2.33, 6.66, 9.99])

print(t.device)  # 我笔电没有支持 CUDA 的 GPU, 就先不测试 t.gpu() 了.
print(t.shape)
print(t.ndim)  # 返回 int 类型
print(tf.rank(t))  # 返回 Tensor 类型
print(t.numpy())

print("----------")

print(tf.is_tensor(1.))
print(tf.is_tensor(np.arange(5)))
print(tf.is_tensor(t))

print("----------")

# Tensor 的转换

a = np.arange(5)
t = tf.convert_to_tensor(a)
print(t)  # 默认 dtype 是 int32
t = tf.convert_to_tensor(a, dtype=tf.int64)
print(t)

# 可以用 tf.cast() 转换数据类型
a = tf.constant([1, 2, 3])
t = tf.cast(a, dtype=tf.float32)
print(t)
t = tf.cast(a, dtype=tf.double)
print(t)

# 0 和 1 可以与 bool 型相互转换
a = tf.constant([False, True, False])
b = tf.cast(a, dtype=tf.int32)
print(b)
print(tf.cast(b, dtype=tf.bool))

print("----------")

# tf Variable
a = tf.range(5)  # 结果是[0, 1, 2, 3, 4] 类似于 np.arange(5)
b = tf.Variable(a)

try:
    print(a.trainable)  # 直接报错
except Exception as e:
    print(e)
print(b.trainable)

print(isinstance(b, tf.Tensor))  # b 是包装过的 Tensor, 此时 isinstance 判断出错
print(tf.is_tensor(b))
