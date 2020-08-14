from builtins import range
from builtins import object
import numpy as np

from cs231n.layers import *
from cs231n.layer_utils import *

class TwoLayerNet(object):
    """
    A two-layer fully-connected neural network with ReLU nonlinearity and
    softmax loss that uses a modular layer design. We assume an input dimension
    of D, a hidden dimension of H, and perform classification over C classes.

    The architecure should be affine - relu - affine - softmax.

    Note that this class does not implement gradient descent; instead, it
    will interact with a separate Solver object that is responsible for running
    optimization.

    The learnable parameters of the model are stored in the dictionary
    self.params that maps parameter names to numpy arrays.
    """

    def __init__(self, input_dim=3*32*32, hidden_dim=100, num_classes=10,
                 weight_scale=1e-3, reg=0.0):
        """
        Initialize a new network.

        Inputs:
        - input_dim: An integer giving the size of the input
        - hidden_dim: An integer giving the size of the hidden layer
        - num_classes: An integer giving the number of classes to classify
        - weight_scale: Scalar giving the standard deviation for random
          initialization of the weights.
        - reg: Scalar giving L2 regularization strength.
        """
        self.params = {}
        self.reg = reg

        ############################################################################
        # TODO: Initialize the weights and biases of the two-layer net. Weights    #
        # should be initialized from a Gaussian centered at 0.0 with               #
        # standard deviation equal to weight_scale, and biases should be           #
        # initialized to zero. All weights and biases should be stored in the      #
        # dictionary self.params, with first layer weights                         #
        # and biases using the keys 'W1' and 'b1' and second layer                 #
        # weights and biases using the keys 'W2' and 'b2'.                         #
        ############################################################################
        self.params["W1"] = weight_scale * np.random.randn(input_dim, hidden_dim)
        self.params["b1"] = np.zeros((hidden_dim))

        self.params["W2"] = weight_scale * np.random.randn(hidden_dim, num_classes )
        self.params["b2"] = np.zeros((num_classes))
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################


    def loss(self, X, y=None):
        """
        Compute loss and gradient for a minibatch of data.

        Inputs:
        - X: Array of input data of shape (N, d_1, ..., d_k)
        - y: Array of labels, of shape (N,). y[i] gives the label for X[i].

        Returns:
        If y is None, then run a test-time forward pass of the model and return:
        - scores: Array of shape (N, C) giving classification scores, where
          scores[i, c] is the classification score for X[i] and class c.

        If y is not None, then run a training-time forward and backward pass and
        return a tuple of:
        - loss: Scalar value giving the loss
        - grads: Dictionary with the same keys as self.params, mapping parameter
          names to gradients of the loss with respect to those parameters.
        """
        scores = None
        ############################################################################
        # TODO: Implement the forward pass for the two-layer net, computing the    #
        # class scores for X and storing them in the scores variable.              #
        ############################################################################
        W1, b1 = self.params['W1'], self.params['b1']
        W2, b2 = self.params['W2'], self.params['b2']

        h, affine_relu_cache = affine_relu_forward(X, W1, b1)
        scores, affine_cache = affine_forward(h, W2, b2)

        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        # If y is None then we are in test mode so just return scores
        if y is None:
            return scores

        loss, grads = 0, {}
        ############################################################################
        # TODO: Implement the backward pass for the two-layer net. Store the loss  #
        # in the loss variable and gradients in the grads dictionary. Compute data #
        # loss using softmax, and make sure that grads[k] holds the gradients for  #
        # self.params[k]. Don't forget to add L2 regularization!                   #
        #                                                                          #
        # NOTE: To ensure that your implementation matches ours and you pass the   #
        # automated tests, make sure that your L2 regularization includes a factor #
        # of 0.5 to simplify the expression for the gradient.                      #
        ############################################################################
        reg = self.reg
        data_loss, dscores = softmax_loss(scores, y)

        reg_loss = 0.5 * reg * np.sum(W1*W1) + 0.5 * reg * np.sum(W2*W2)
        loss = data_loss + reg_loss

        dh, dW2, db2 = affine_backward(dscores, affine_cache)
        dx, dW1, db1 = affine_relu_backward(dh, affine_relu_cache)

        # add regularization gradient contribution
        dW1 +=  reg * W1
        dW2 +=  reg * W2

        grads = {"W1":dW1, "W2":dW2, "b1":db1, "b2":db2}

        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        return loss, grads


class FullyConnectedNet(object):
    """
    A fully-connected neural network with an arbitrary number of hidden layers,
    ReLU nonlinearities, and a softmax loss function. This will also implement
    dropout and batch/layer normalization as options. For a network with L layers,
    the architecture will be

    {affine - [batch/layer norm] - relu - [dropout]} x (L - 1) - affine - softmax

    where batch/layer normalization and dropout are optional, and the {...} block is
    repeated L - 1 times.

    Similar to the TwoLayerNet above, learnable parameters are stored in the
    self.params dictionary and will be learned using the Solver class.
    """

    def __init__(self, hidden_dims, input_dim=3*32*32, num_classes=10,
                 dropout=1, normalization=None, reg=0.0,
                 weight_scale=1e-2, dtype=np.float32, seed=None):
        """
        Initialize a new FullyConnectedNet.

        Inputs:
        - hidden_dims: A list of integers giving the size of each hidden layer.
        - input_dim: An integer giving the size of the input.
        - num_classes: An integer giving the number of classes to classify.
        - dropout: Scalar between 0 and 1 giving dropout strength. If dropout=1 then
          the network should not use dropout at all.
        - normalization: What type of normalization the network should use. Valid values
          are "batchnorm", "layernorm", or None for no normalization (the default).
        - reg: Scalar giving L2 regularization strength.
        - weight_scale: Scalar giving the standard deviation for random
          initialization of the weights.
        - dtype: A numpy datatype object; all computations will be performed using
          this datatype. float32 is faster but less accurate, so you should use
          float64 for numeric gradient checking.
        - seed: If not None, then pass this random seed to the dropout layers. This
          will make the dropout layers deteriminstic so we can gradient check the
          model.
        """
        self.normalization = normalization
        self.use_dropout = dropout != 1
        self.reg = reg
        self.num_layers = 1 + len(hidden_dims)
        self.dtype = dtype
        self.params = {}

        ############################################################################
        # TODO: Initialize the parameters of the network, storing all values in    #
        # the self.params dictionary. Store weights and biases for the first layer #
        # in W1 and b1; for the second layer use W2 and b2, etc. Weights should be #
        # initialized from a normal distribution centered at 0 with standard       #
        # deviation equal to weight_scale. Biases should be initialized to zero.   #
        #                                                                          #
        # When using batch normalization, store scale and shift parameters for the #
        # first layer in gamma1 and beta1; for the second layer use gamma2 and     #
        # beta2, etc. Scale parameters should be initialized to ones and shift     #
        # parameters should be initialized to zeros.                               #
        ############################################################################
        layer_dims = [input_dim] + hidden_dims + [num_classes]

        for layer_num in range(self.num_layers):
            self.params["W{}".format(layer_num+1)] =  \
            weight_scale *  np.random.randn(layer_dims[layer_num],
                                        layer_dims[layer_num + 1])
            self.params["b{}".format(layer_num+1)] = \
                                        np.zeros((layer_dims[layer_num + 1]))

        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        # When using dropout we need to pass a dropout_param dictionary to each
        # dropout layer so that the layer knows the dropout probability and the mode
        # (train / test). You can pass the same dropout_param to each dropout layer.
        self.dropout_param = {}
        if self.use_dropout:
            self.dropout_param = {'mode': 'train', 'p': dropout}
            if seed is not None:
                self.dropout_param['seed'] = seed

        # With batch normalization we need to keep track of running means and
        # variances, so we need to pass a special bn_param object to each batch
        # normalization layer. You should pass self.bn_params[0] to the forward pass
        # of the first batch normalization layer, self.bn_params[1] to the forward
        # pass of the second batch normalization layer, etc.
        self.bn_params = []
        if self.normalization=='batchnorm':
            self.bn_params = [{'mode': 'train'} for i in range(self.num_layers - 1)]
            for i in range(1, self.num_layers):
                D = hidden_dims[i-1]
                self.params['gamma{}'.format(i)] = np.ones(D)
                self.params['beta{}'.format(i)] = np.zeros(D)

        if self.normalization=='layernorm':
            self.ln_params = [{} for i in range(self.num_layers - 1)]
            for i in range(1, self.num_layers):
                D = hidden_dims[i-1]
                self.params['gamma{}'.format(i)] = np.ones(D)
                self.params['beta{}'.format(i)] = np.zeros(D)

        # Cast all parameters to the correct datatype
        for k, v in self.params.items():
            self.params[k] = v.astype(dtype)


    def loss(self, X, y=None):
        """
        Compute loss and gradient for the fully-connected net.

        Input / output: Same as TwoLayerNet above.
        """
        X = X.astype(self.dtype)
        mode = 'test' if y is None else 'train'

        # Set train/test mode for batchnorm params and dropout param since they
        # behave differently during training and testing.
        if self.use_dropout:
            self.dropout_param['mode'] = mode
        if self.normalization=='batchnorm':
            for bn_param in self.bn_params:
                bn_param['mode'] = mode

        scores = None
        ############################################################################
        # TODO: Implement the forward pass for the fully-connected net, computing  #
        # the class scores for X and storing them in the scores variable.          #
        #                                                                          #
        # When using dropout, you'll need to pass self.dropout_param to each       #
        # dropout forward pass.                                                    #
        #                                                                          #
        # When using batch normalization, you'll need to pass self.bn_params[0] to #
        # the forward pass for the first batch normalization layer, pass           #
        # self.bn_params[1] to the forward pass for the second batch normalization #
        # layer, etc.                                                              #
        ############################################################################


        caches = {}  #caches[i] = cache for backprop of layer i

        for i in range(1, self.num_layers):

            # Affine forward
            Wi = self.params["W{}".format(i)]
            bi = self.params["b{}".format(i)]
            X, cache = affine_forward(X, Wi, bi)
            caches["affine_forward{}".format(i)] = cache

            # Batchnorm forward (optional)
            if self.normalization=='batchnorm':
                gamma_i = self.params['gamma{}'.format(i)]
                beta_i = self.params['beta{}'.format(i)]
                X, cache = batchnorm_forward(X, gamma_i, beta_i, self.bn_params[i-1])
                caches['batchnorm_forward{}'.format(i)] = cache

            # Layernorm forward (optional)
            if self.normalization=='layernorm':
                gamma_i = self.params['gamma{}'.format(i)]
                beta_i = self.params['beta{}'.format(i)]
                X, cache = layernorm_forward(X, gamma_i, beta_i, self.ln_params[i-1])
                caches['layernorm_forward{}'.format(i)] = cache

            # Relu forward
            X, cache = relu_forward(X)
            caches["relu_forward{}".format(i)] = cache

            # Dropout forward (optional)
            if self.use_dropout:
                X, cache = dropout_forward(X, self.dropout_param)
                caches["dropout_forward{}".format(i)] = cache

        # Final layer forward
        scores, cache = affine_forward(X, self.params["W{}".format(i+1)],
                               self.params["b{}".format(i+1)])
        caches["affine_forward{}".format(i+1)] = cache

        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        # If test mode return early
        if mode == 'test':
            return scores

        loss, grads = 0.0, {}
        ############################################################################
        # TODO: Implement the backward pass for the fully-connected net. Store the #
        # loss in the loss variable and gradients in the grads dictionary. Compute #
        # data loss using softmax, and make sure that grads[k] holds the gradients #
        # for self.params[k]. Don't forget to add L2 regularization!               #
        #                                                                          #
        # When using batch/layer normalization, you don't need to regularize the scale   #
        # and shift parameters.                                                    #
        #                                                                          #
        # NOTE: To ensure that your implementation matches ours and you pass the   #
        # automated tests, make sure that your L2 regularization includes a factor #
        # of 0.5 to simplify the expression for the gradient.                      #
        ############################################################################
        loss, grads = 0, {}

        reg = self.reg
        data_loss, dscores = softmax_loss(scores, y)

        # add regularization loss
        reg_loss = 0
        for i in range(1, self.num_layers + 1):
            Wi = self.params["W{}".format(i)]
            reg_loss += 0.5 * reg * np.sum(Wi*Wi)

        loss = data_loss + reg_loss


        #backprob of final affine forward (no ReLu)
        dfinal_hidden_layer, dW, db = affine_backward(dscores,
                      caches["affine_forward{}".format(self.num_layers)])

        grads["W{}".format(self.num_layers)] = dW
        grads["b{}".format(self.num_layers)] = db

        # backprop of first L-1 layers
        #You might find it useful to define an additional helper layer similar to those
        # in the file cs231n/layer_utils.py. If you decide to do so, do it in the file
        # cs231n/classifiers/fc_net.py.
        dout = dfinal_hidden_layer
        for i in range(self.num_layers - 1, 0, -1):

            if self.use_dropout:
                # backprop dropout
                cache = caches['dropout_forward{}'.format(i)]
                dout = dropout_backward(dout, cache)

            # backprop ReLu
            cache = caches["relu_forward{}".format(i)]
            dx = relu_backward(dout, cache)

            if self.normalization=='batchnorm':
                # backprop Batchnorm
                cache = caches['batchnorm_forward{}'.format(i)]
                dx, dgamma, dbeta = batchnorm_backward(dx, cache)
                grads["gamma{}".format(i)] = dgamma
                grads["beta{}".format(i)] = dbeta

            if self.normalization=='layernorm':
                # backprop layernorm
                cache = caches['layernorm_forward{}'.format(i)]
                dx, dgamma, dbeta = layernorm_backward(dx, cache)
                grads["gamma{}".format(i)] = dgamma
                grads["beta{}".format(i)] = dbeta

            # backprop affine
            cache = caches["affine_forward{}".format(i)]
            dout, dWi, dbi = affine_backward(dx, cache)

            # store gradient of parameters
            grads["W{}".format(i)] = dWi
            grads["b{}".format(i)] = dbi

        # add regularization gradient contribution
        for i in range(1, self.num_layers + 1):
            Wi = self.params["W{}".format(i)]
            grads["W{}".format(i)] += reg * Wi

        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################
        return loss, grads
