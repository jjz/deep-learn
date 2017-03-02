from gen_auth_code import *;

import numpy
import tensorflow as tf

auth, image = gen_auth_image()

IMAGE_HEIGHT = 60
IMAGE_WIDTH = 160
MAX_CAPTCHA = len(auth)


def conver2gray(img):
    if len(img.shape) > 2:
        gray = numpy.mean(img, -1)
        return gray
    else:
        return img


char_set = number + alphabet + ALPHABET + ['_']
CHAR_SET_LEN = len(char_set)


def text2vec(text):
    text_len = len(text)
    if text_len > MAX_CAPTCHA:
        raise ValueError('legth >4')
    vector = numpy.zeros(MAX_CAPTCHA * CHAR_SET_LEN)
    for i, c in enumerate(text):
        idx = i * CHAR_SET_LEN + char2pos(c)
        vector[idx] = 1
    return vector


def char2pos(c):
    if c == '_':
        k = 62
        return k
    k = ord(c) - 48
    if k > 9:
        k = ord(c) - 55
        if k > 35:
            k = ord(c) - 61
            if k > 61:
                raise ValueError('no map')
    return k


def vec2text(vec):
    char_pos = vec.nonzero()[0]
    text = []
    for i, c in enumerate(char_pos):
        char_at_pos = i
        char_idx = c % CHAR_SET_LEN
        if char_idx < 10:
            char_code = char_idx + ord('0')
        elif char_idx < 36:
            char_code = char_idx - 10 + ord('A')
        elif char_idx == 62:
            char_code = ord('_')
        else:
            raise ValueError('error')
        text.append(chr(char_code))
    return "".join(text)


def wrap_gen_auth_text_and_image():
    while True:
        text, img = gen_auth_image()
        if img.shape == (60, 160, 3):
            return text, image


def get_next_batch(batch_size=128):
    batch_x = numpy.zeros([batch_size, IMAGE_HEIGHT * IMAGE_WIDTH])
    batch_y = numpy.zeros([batch_size, MAX_CAPTCHA * CHAR_SET_LEN])
    for i in range(batch_size):
        text, img = wrap_gen_auth_text_and_image()
        image = conver2gray(img)

        batch_x[i, :] = image.flatten()
        batch_y[i, :] = text2vec(text)
    return batch_x, batch_y


X = tf.placeholder(tf.float32, [None, IMAGE_HEIGHT * IMAGE_WIDTH])
Y = tf.placeholder(tf.float32, [None, MAX_CAPTCHA * CHAR_SET_LEN])
keep_prob = tf.placeholder(tf.float32)


def crack_auth_cnn(w_alpha=0.01, b_alpha=0.1):
    x = tf.reshape(X, shape=[-1, IMAGE_HEIGHT, IMAGE_WIDTH, 1])

    w_cl = tf.Variable(w_alpha * tf.random_normal([3, 3, 1, 32]))
    b_cl = tf.Variable(b_alpha * tf.random_normal([32]))
    conv1 = tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(x, w_cl, strides=[1, 1, 1, 1], padding='SAME'), b_cl))
    conv1 = tf.nn.max_pool(conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    conv1 = tf.nn.dropout(conv1, keep_prob)

    w_c2 = tf.Variable(w_alpha * tf.random_normal([3, 3, 32, 64]))
    b_c2 = tf.Variable(b_alpha * tf.random_normal([64]))
    conv2 = tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(conv1, w_c2, strides=[1, 1, 1, 1], padding='SAME'), b_c2))
    conv2 = tf.nn.max_pool(conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    conv2 = tf.nn.dropout(conv2, keep_prob)

    w_c3 = tf.Variable(w_alpha * tf.random_normal([3, 3, 64, 64]))
    b_c3 = tf.Variable(b_alpha * tf.random_normal([64]))
    conv3 = tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(conv2, w_c3, strides=[1, 1, 1, 1], padding='SAME'), b_c3))
    conv3 = tf.nn.max_pool(conv3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    conv3 = tf.nn.dropout(conv3, keep_prob)

    w_d = tf.Variable(w_alpha * tf.random_normal([8 * 32 * 40, 1024]))
    b_d = tf.Variable(b_alpha * tf.random_normal([1024]))
    dense = tf.reshape(conv3, [-1, w_d.get_shape().as_list()[0]])
    dense = tf.nn.relu(tf.add(tf.matmul(dense, w_d), b_d))
    dense = tf.nn.dropout(dense, keep_prob)

    w_out = tf.Variable(w_alpha * tf.random_normal([1024, MAX_CAPTCHA * CHAR_SET_LEN]))
    b_out = tf.Variable(b_alpha * tf.random_normal([MAX_CAPTCHA * CHAR_SET_LEN]))
    out = tf.add(tf.matmul(dense, w_out), b_out)
    return out


def train_crack_auth_cnn():
    output = crack_auth_cnn()
    loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=output, logits=Y))
    optimizer = tf.train.AdamOptimizer(learning_rate=0.001).minimize(loss)
    predict = tf.reshape(output, [-1, MAX_CAPTCHA, CHAR_SET_LEN])
    max_idx_p = tf.argmax(predict, 2)
    max_idx_1 = tf.argmax(tf.reshape(Y, [-1, MAX_CAPTCHA, CHAR_SET_LEN]), 2)
    correct_pred = tf.equal(max_idx_p, max_idx_1)
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
    saver = tf.train.Saver()
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        setp = 0
        while True:
            batch_x, batch_y = get_next_batch(64)
            _, loss_ = sess.run([optimizer, loss], feed_dict={X: batch_x, Y: batch_y, keep_prob: 0.75})
            print("loss:{},{}".format(setp, loss_))
            if setp % 100 == 0:
                batch_x_test, batch_y_test = get_next_batch(100)
                acc = sess.run(accuracy, feed_dict={X: batch_x_test, Y: batch_y_test, keep_prob: 1.})
                print("acc:{},{}".format(setp, acc))
                if acc > 0.5:
                    saver.save(sess, "crack_capcha.model", global_step=setp)
            setp += 1


train_crack_auth_cnn()
