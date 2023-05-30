import numpy as np
import sklearn
assert sklearn.__version__>="0.20"
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt

np.random.seed(42)
tf.random.set_seed(42)

def generate_3d_data(m,w1=0.1,w2=0.3,noise=0.1):
    angles=np.random.rand(m)*3*np.pi/2-0.5
    data=np.empty((m,3))
    data[:,0]=np.cos(angles)+np.sin(angles)/2+noise*np.random(m)/2
    data[:,1]=np.sin(angles)*0.7+noise*np.random.randn(m)/2
    data[:,2]=data[:,0]*w1+data[:,1]*w2+noise*np.random.randn(m)
    return data

X_train=generate_3d_data(60)
X_train=X_train-X_train.mean(axis=0,keepdims=0)