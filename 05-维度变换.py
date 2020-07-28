import tensorflow as tf

# 改变View (Content 即数据本身不变)

# 比如将 MNIST 中的[b, 28, 28] 转换为 [b, 28 * 28], 后者中行和列之间的关系消失了
# 也可进一步将图片分为两个部分, 即[b, 2, 14 * 28]
# 也可增加通道 [b, 28, 28, 1]
# 每一种表示方法即为一种 View

# tf.reshape
a = tf.random.normal([4, 28, 28, 3])
print(tf.reshape(a, [4, 784, 3]).shape)  # 失去二维信息
print(tf.reshape(a, [4, -1, 3]).shape)  # 可填 -1 让系统自动计算

try:
    print(tf.reshape(a, [4, 784 - 1, 3]))  # 报错,总数不一致
except Exception as e:
    print(e)

print(tf.reshape(a, [4, 28 * 28 * 3]).shape)  # 将 channel 信息也合并
print(tf.reshape(a, [4, -1]).shape)  # 与上一行等价

# reshape 后也可恢复
# 但 reshape 之后需要确保其本身意义不变
# 比如 height 28, width 28 的图片直接 reshape 为 width 28, height 28 的图片, 虽然 shape 相同, 但含义已经改变
# 必须使用 Transpose
print(tf.reshape(tf.reshape(a, [4, -1]), [4, 28, 28, 3]).shape)

print("----------")

# Transpose (改变 Content)
# 例: 将 [b, h, w, c] transpose 成 [b, w, h, c] 使用 Transpose 使其有意义
a = tf.random.normal([4, 3, 2, 1])

print(tf.transpose(a).shape)  # 默认将其直接倒转
print(tf.transpose(a, perm=[0, 1, 3, 2]).shape)  # 只将后两个维度倒转

# 例: 将 [b, 3, h, w] 转换为 [b, h, w, 3]
a = tf.random.normal([10, 3, 28, 28])  # [b, 3, h, w]
print(a.shape)
print(tf.transpose(a, perm=[0, 3, 2, 1]).shape)  # 错误: 结果是 [b, w, h, 3]
print(tf.transpose(a, perm=[0, 2, 3, 1]).shape)  # 正确: 结果是 [b, h, w, 3]

