import tensorflow as tf
from model import *
from gen_auth_code import *


def check_auth(auth_image):
    output = crack_auth_cnn()
    saver = tf.train.Saver()
    with tf.Session() as sess:
        saver.restore(sess, tf.train.latest_checkpoint('.'))
        predict = tf.argmax(tf.reshape(output, [-1, MAX_CAPTCHA, CHAR_SET_LEN]), 2)
        text_list = sess.run(predict, feed_dict={X: [auth_image], keep_prob: 1})
        text = text_list[0].tolist()
        return text


text, image = gen_auth_image()
image = conver2gray(image)
image = image.flatten() / 255
predict_text = check_auth(image)
print ("auth:{} train:{}".format(text, predict_text))
