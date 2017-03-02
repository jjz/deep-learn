import numpy as np
import tensorflow as tf

x_data = np.random.rand(100).astype(np.float32)
y_data = x_data * 0.1 + 0.3

w = tf.Variable(tf.random_uniform([1], -1.0, 1.0))
b = tf.Variable(tf.zeros([1]))
y = w * x_data + b

loss =tf.reduce_mean(tf.square(y-y_data))
optimize =tf.train.GradientDescentOptimizer(0.05)
train =optimize.minimize(loss)

init =tf.global_variables_initializer()

sess =tf.Session()
sess.run(init)

for step in range(1601):
    sess.run(train)
    if step%10==0:
        print(step,sess.run(w),sess.run(b))

sess.close()
