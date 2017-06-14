import tensorflow as tf
import gzip
import pickle
from tensorflow.examples.tutorials.mnist import input_data

def load_mnist():
    with gzip.open('MNIST_data/mnist.pkl.gz', 'rb') as f:
        train_set, valid_set, test_set = pickle.load(f, encoding='latin1')
        return train_set, valid_set, test_set

class MNIST:
    def __init__(self):
        self.g = tf.Graph()
        with self.g .as_default():
            with tf.variable_scope("input"):
                self.x = tf.placeholder(tf.float32, shape=[None, 784], name='x-input')
                self.y_ = tf.placeholder(tf.float32, shape=[None, 10], name='y-input')
                with tf.variable_scope("input_reshape"):
                    self.image_shaped_input = tf.reshape(x, [-1, 28, 28, 1)]tf
                    tf.summary(image('input', image_shaped_input, 10)
            
        
        
    
if __name__ == "__main__":
    #train_set, train_set, valid_set = load_mnist()
    #train_x, train_y = train_set
    #print(train_x[0])
    #print(train_x[0].shape)
    #print(train_x.shape)
    #print(train_y.shape)

    
    
    
