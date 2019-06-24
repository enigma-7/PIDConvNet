import tensorflow as tf
import numpy as np
import matplotlib
matplotlib.rcParams.update({'font.size': 18})
matplotlib.rcParams['text.usetex'] = True


tf.enable_eager_execution()

class Model(object):
    def __init__(self):
        # Initialize variable to (5.0, 0.0)
        # In practice, these should be initialized to random values.
        self.W = tf.Variable(5.0)
        self.b = tf.Variable(0.0)

    def __call__(self, x):
        return self.W * x + self.b

model = Model()

def loss(predicted_y, desired_y):
    return tf.reduce_mean(tf.square(predicted_y - desired_y))         #sum((y_- y)^2)/len(y)

TRUE_W = 3.0
TRUE_b = 2.0
NUM_EXAMPLES = 1000

inputs  = tf.random_normal(shape=[NUM_EXAMPLES])            #Values from a gaussian
noise   = tf.random_normal(shape=[NUM_EXAMPLES])
outputs = inputs * TRUE_W + TRUE_b + noise
print(model(inputs))
import matplotlib.pyplot as plt

plt.figure(figsize=(8,6))
plt.scatter(inputs, outputs, c='b', label='Given')
plt.plot(inputs, model(inputs), c='r', label='Model')
plt.legend()
plt.grid()
plt.title('Initial Unlearned Estimation')
plt.show()

print('Current loss: %.2f '%(loss(model(inputs), outputs).numpy()))

def train(model, inputs, outputs, learning_rate):
    with tf.GradientTape() as t:
        current_loss = loss(model(inputs), outputs)             #cost function
    dW, db = t.gradient(current_loss, [model.W, model.b])       #evaluate partial differentials !!!!
    model.W.assign_sub(learning_rate * dW)                      #Learning weights
    model.b.assign_sub(learning_rate * db)                      #Learning biases

# Collect the history of W-values and b-values to plot later
Ws, bs = [], []
epochs = range(40)
for epoch in epochs:
  Ws.append(model.W.numpy())
  bs.append(model.b.numpy())
  current_loss = loss(model(inputs), outputs)

  train(model, inputs, outputs, learning_rate=0.1)
  print('Epoch %2d: W=%1.2f b=%1.2f, loss=%2.5f' %
        (epoch, Ws[-1], bs[-1], current_loss))

# Let's plot it all
plt.figure(figsize=(8,6))
plt.plot(epochs, Ws, 'r',
         epochs, bs, 'b')
plt.plot([TRUE_W] * len(epochs), 'r--',
         [TRUE_b] * len(epochs), 'b--')
plt.legend(['$W$', '$b$', '$W_{true}$', '$b_{true}$'])
plt.show()

plt.figure(figsize=(8,6))
plt.scatter(inputs, outputs, c='b', label='Given')
plt.plot(inputs, model(inputs), c='r', label='Model')
plt.legend()
plt.grid()
plt.show()
