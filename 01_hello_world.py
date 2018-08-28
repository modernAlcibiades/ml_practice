import tensorflow as tf

def op():
    x = tf.placeholder(shape=None, dtype=tf.float32, name='x')
    y = tf.placeholder(shape=None, dtype=tf.float32, name='y')
    z = tf.placeholder(shape=None, dtype=tf.float32, name='z')
    add_op = tf.add(x,y)
    mul_op = tf.multiply(x,add_op)
    power_op = tf.pow(mul_op, z)
    return power_op


if __name__=="__main__":
    feed_dict ={'x:0':2, 'y:0':4, 'z:0':5}
    op = op()
    with tf.Session() as sess:
        ans = sess.run(op, feed_dict = feed_dict)
        writer = tf.summary.FileWriter("./logs/", sess.graph)
        writer.close()
    print('(x*(x+y))^z given ',feed_dict, ans)
