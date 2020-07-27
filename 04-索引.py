import tensorflow as tf

# 最经典的索引方式
a = tf.random.normal([3, 5, 5, 4])
print(a[0])
print(a[0][0])
print(a[0][0][0])
print(a[0][0][0][0])

print("----------")

# numpy 风格索引 (与上方索引方法一一对应)
print(a[0])
print(a[0, 0])
print(a[0, 0, 0])
print(a[0, 0, 0, 0])

print("----------")

# 切片 (与 python list 的切片方式类似)
a = tf.range(10)

# [A:B] 代表切片 [A, B) 范围
print(a[0:2])
print(a[2:])
print(a[-1:])  # 切片返回的始终是 vector, 即使只有一个元素
print(a[-2:])

# 高维度
a = tf.random.uniform([5, 4, 3, 2])
print(a.shape)
print(a[0, :, :, :])  # 等价于 a[0], 单独的冒号表示全取
print(a[0, 1, :, :])  # 等价于 a[0, 1]
print(a[:, :, :, 0])  # 在只取中间或最后时必须需要冒号

print("----------")

# 双冒号: 隔行切片/采样 [start:end:step]
a = tf.range(20)
print(a)
print(a[0:5:2])  # 从下标 0 到下标 5 , 一隔一采样
print(a[::2])  # start 和 end 全部省略
print(a[::3])  # start 和 end 全部省略

# 高纬度同理

# 双冒号可以实现逆序
print(a[::-1])  # 即 step = -1, 即倒序
print(a[::-2])  # 倒着隔行采样

print("----------")

# 省略号: 代表任意长度冒号
a = tf.random.uniform([2, 3, 4, 5, 6])
print(a[0, :, :, :, :].shape)
print(a[0, ...].shape)  # 两种方式等价, 用省略号更优雅
print(a[1, ..., 3].shape)  # 用在中间也行
print(a[1, ..., 3, :].shape)

try:
    print(a[0, ..., 1, ...])  # 无法推断报错
except Exception as e:
    print(e)