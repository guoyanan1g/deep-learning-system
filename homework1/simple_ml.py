"""hw1/apps/simple_ml.py"""

import struct
import gzip
import numpy as np

import sys

sys.path.append("python/")
import needle as ndl


def parse_mnist(image_filesname, label_filename):
    """Read an images and labels file in MNIST format.  See this page:
    http://yann.lecun.com/exdb/mnist/ for a description of the file format.

    Args:
        image_filename (str): name of gzipped images file in MNIST format
        label_filename (str): name of gzipped labels file in MNIST format

    Returns:
        Tuple (X,y):
            X (numpy.ndarray[np.float32]): 2D numpy array containing the loaded
                data.  The dimensionality of the data should be
                (num_examples x input_dim) where 'input_dim' is the full
                dimension of the data, e.g., since MNIST images are 28x28, it
                will be 784.  Values should be of type np.float32, and the data
                should be normalized to have a minimum value of 0.0 and a
                maximum value of 1.0.

            y (numpy.ndarray[dypte=np.int8]): 1D numpy array containing the
                labels of the examples.  Values should be of type np.int8 and
                for MNIST will contain the values 0-9.
    """
    ### BEGIN YOUR SOLUTION
    with gzip.open(image_filesname,'rb') as f:
      #size 11760004 can not divide 784
      #/255是为了归一化，缩放至0-1
      # MNIST 数据集的图像文件的前 16 字节包含了一些元数据，如魔法数（magic number）、图像数量、行数和列数等信息，而不是像素数据。
      #标签文件前8字节也是元数据
      #读取的时候以uint8为单位，uint8可以表示0-255
      X=np.frombuffer(f.read(),dtype=np.uint8,count=-1,offset=16).reshape(-1,784).astype('float32')/255
    with gzip.open(label_filename) as l:
      y=np.frombuffer(l.read(),dtype=np.uint8,count=-1,offset=8)
    return (X,y)
    ### END YOUR SOLUTION


def one_hot_decode(y):
  for i in len(y):
    if y[i]==1:
      return i+1
def softmax_loss(Z, y_one_hot):
    """Return softmax loss.  Note that for the purposes of this assignment,
    you don't need to worry about "nicely" scaling the numerical properties
    of the log-sum-exp computation, but can just compute this directly.

    Args:
        Z (ndl.Tensor[np.float32]): 2D Tensor of shape
            (batch_size, num_classes), containing the logit predictions for
            each class.
        y (ndl.Tensor[np.int8]): 2D Tensor of shape (batch_size, num_classes)
            containing a 1 at the index of the true label of each example and
            zeros elsewhere.

    Returns:
        Average softmax loss over the sample. (ndl.Tensor[np.float32])
    """
    ### BEGIN YOUR SOLUTION
    exp_sum_z=ndl.summation(ndl.exp(Z),axes=(-1,))
    b=Z.shape[0]
    z_y=ndl.summation(ndl.multiply(Z,y_one_hot),axes=(-1,))#multiply:元素乘
    return ndl.summation((ndl.log(exp_sum_z)-z_y),axes=(-1,))/b
    ### END YOUR SOLUTION


def nn_epoch(X, y, W1, W2, lr=0.1, batch=100):
    """Run a single epoch of SGD for a two-layer neural network defined by the
    weights W1 and W2 (with no bias terms):
        logits = ReLU(X * W1) * W1
    The function should use the step size lr, and the specified batch size (and
    again, without randomizing the order of X).

    Args:
        X (np.ndarray[np.float32]): 2D input array of size
            (num_examples x input_dim).
        y (np.ndarray[np.uint8]): 1D class label array of size (num_examples,)
        W1 (ndl.Tensor[np.float32]): 2D array of first layer weights, of shape
            (input_dim, hidden_dim)
        W2 (ndl.Tensor[np.float32]): 2D array of second layer weights, of shape
            (hidden_dim, num_classes)
        lr (float): step size (learning rate) for SGD
        batch (int): size of SGD mini-batch

    Returns:
        Tuple: (W1, W2)
            W1: ndl.Tensor[np.float32]
            W2: ndl.Tensor[np.float32]
    """

    ### BEGIN YOUR SOLUTION
    n=X.shape[0]
    m=W2.shape[-1]
    one_hot_y=np.zeros((n,m))
    one_hot_y[np.arange(n),y]=1
    times=n//batch
    for i in range(times+1):
      start=i*batch
      end=min(n,start+batch)
      if start==end:
        break
      x=ndl.Tensor(X[start:end])
      yy=ndl.Tensor(one_hot_y[start:end])
      tmp=ndl.relu(ndl.matmul(x,W1)) 
      Z=ndl.matmul(tmp,W2)
      loss=softmax_loss(Z,yy)
      loss.backward()
      grad1=W1.grad.numpy()
      grad2=W2.grad.numpy()
      #.numpy() is easier to compute
      W1=ndl.Tensor(W1.numpy()-lr*grad1)
      W2=ndl.Tensor(W2.numpy()-lr*grad2)
    return W1,W2
    ### END YOUR SOLUTION


### CODE BELOW IS FOR ILLUSTRATION, YOU DO NOT NEED TO EDIT


def loss_err(h, y):
    """Helper function to compute both loss and error"""
    y_one_hot = np.zeros((y.shape[0], h.shape[-1]))
    y_one_hot[np.arange(y.size), y] = 1
    y_ = ndl.Tensor(y_one_hot)
    return softmax_loss(h, y_).numpy(), np.mean(h.numpy().argmax(axis=1) != y)
