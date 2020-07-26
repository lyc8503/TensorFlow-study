import tensorflow as tf

# scalar: loss, accuracy

out = tf.random.uniform([4, 10])
print(out)

y = tf.range(4)
y = tf.one_hot(y, depth=10)
print(y)

loss = tf.keras.losses.mse(y, out)
loss = tf.reduce_mean(loss)

print(loss)

print("----------")

# vector: bias / matrix: input / weight
net = tf.keras.layers.Dense(10)
net.build((4, 8))
print(net.kernel)  # matrix
print(net.bias)  # vector

# dim = 3 Tensor 例: 自然语言处理

# dim = 4 Tensor 例: 图片 [b(图片个数), h(长), w(宽), c(通道数)]

# dim = 5 Tensor 例: meta-learning
