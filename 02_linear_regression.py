import tensorflow as tf
import numpy as np
import time
#### inference: y_pred = w*X +b
### mse : E[y-y_pred)^2]

def read_birth_life_data(filename='data/birth_life_2010.txt'):
    text = open(filename, 'r').readlines()[1:]
    data = [line[:-1].split('\t') for line in text[:-1]]
    #for line in data:
    #    print(line[1])
    births = [float(line[1]) for line in data]
    lifes = [float(line[2]) for line in data]
    data = list(zip(births,lifes))
    n_samples = len(data)
    data = np.asarray(data, dtype=np.float32)
    return data, n_samples


def lin_reg_model(num_epochs):
    global_step = tf.Variable(0,trainable=False)
    x = tf.placeholder(dtype=tf.float32, name='x')
    y = tf.placeholder(dtype=tf.float32, name='y')
    lr = tf.train.exponential_decay(0.02, global_step, num_epochs, 0.9, staircase=True)
    #lr = tf.placeholder(dtype=tf.float32, name='lr')

    w = tf.get_variable('weights', initializer=tf.constant(0.0))
    b = tf.get_variable('bias', initializer = tf.constant(0.0))
    y_pred = w*x + b
    loss = tf.square(y - y_pred, name='loss')
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=lr).minimize(loss, global_step = global_step)
    return optimizer, loss

def run(dataset,n_samples):
    #### Dataset
    iterator = dataset.make_initializable_iterator()
    X,Y = iterator.get_next()
    #### Model
    num_epochs = 100
    optimizer, loss = lin_reg_model(num_epochs)

    #### Timing
    start = time.time()
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        writer= tf.summary.FileWriter('./logs/lin_reg', sess.graph)
        for i in range(num_epochs):
            sess.run(iterator.initializer)
            total_loss = 0
            try:
                while True:
                    x,y = sess.run([X,Y])
                    feed_dict = {'x:0':x, 'y:0':y}
                    _, l = sess.run([optimizer,loss], feed_dict=feed_dict)
                    #_, l = sess.run([optimizer,loss])
                    total_loss +=l
            except tf.errors.OutOfRangeError:
                pass
            print('Average loss for Epoch {0}: {1}'.format(i, total_loss/n_samples))
        writer.close()

        w_out, b_out = sess.run([w,b])
        print('w %f, b: %f' %(w_out, b_out))
    print('Total time: %f seconds' %(time.time()-start))

if __name__=="__main__":
    # Create dataset
    data, n_samples = read_birth_life_data()
    dataset = tf.data.Dataset.from_tensor_slices((data[:,0], data[:,1])) #x, y
    run(dataset,n_samples)
