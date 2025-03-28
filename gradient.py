import torch
import tensorflow as tf
import jax.numpy as jnp
from jax import grad

# # use pytorch
x0 = 2.0
# x = torch.tensor(2.0, requires_grad=True) #x for f(x) below
# f = x**3 + 2*x + 1 #f(x)
# f.backward() #f'(x)
# print(x.grad)

# #use tensorflow
# x = tf.Variable(x0)
# with tf.GradientTape() as tape:
#    f = x**3 + 2*x + 1
# dfdx = tape.gradient(f, x)

# print(dfdx.numpy())

#use jax
def f(x):
    return x**3 + 2*x + 1

dfdx = grad(f) #f'(x)
print(dfdx(x0)) #f'(x0)