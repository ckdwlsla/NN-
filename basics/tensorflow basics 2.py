import tensorflow as tf
import numpy as np

def contrib_estimator():
    
    features = [tf.contrib.layers.real_valued_column("x", dimension=1)]

    estimator = tf.contrib.learn.LinearRegressor(feature_columns = features)

    return estimator
    

def custom_contrib_model(features, labels, mode):
    W = tf.get_variable("W", [1], dtype=tf.float64)
    b = tf.get_variable("b", [1], dtype=tf.float64)
    y = W*features['x'] + b
    loss = tf.reduce_sum(tf.square(y- labels))

    global_step = tf.train.get_global_step()
    optimizer = tf.train.GradientDescentOptimizer(0.01)
    train = tf.group(optimizer.minimize(loss), tf.assign_add(global_step, 1))
    return tf.contrib.learn.ModelFnOps(
        mode=mode, predictions=y,
        loss=loss,
        train_op=train)
    


if __name__ == "__main__":
    x = np.array([1., 2., 3., 4.])
    y = np.array([0., -1., -2., -3.])
    input_fn = tf.contrib.learn.io.numpy_input_fn({"x":x}, y, batch_size=4, num_epochs=1000)
    estimator1 = contrib_estimator()
    estimator1.fit(input_fn=input_fn, steps=1000)
    print(estimator1.evaluate(input_fn=input_fn))

    input_fn2 = tf.contrib.learn.io.numpy_input_fn({"x": x}, y, 4, num_epochs=1000)
    estimator2 = tf.contrib.learn.Estimator(model_fn=custom_contrib_model)
    estimator2.fit(input_fn=input_fn2, steps=1000)
    print(estimator2.evaluate(input_fn=input_fn2, steps=10))
    
