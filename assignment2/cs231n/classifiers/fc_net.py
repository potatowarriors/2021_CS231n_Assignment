from builtins import range
from builtins import object
import numpy as np

from ..layers import *
from ..layer_utils import *


class FullyConnectedNet(object):
    """Class for a multi-layer fully connected neural network.

    Network contains an arbitrary number of hidden layers, ReLU nonlinearities,
    and a softmax loss function. This will also implement dropout and batch/layer
    normalization as options. For a network with L layers, the architecture will be

    {affine - [batch/layer norm] - relu - [dropout]} x (L - 1) - affine - softmax

    where batch/layer normalization and dropout are optional and the {...} block is
    repeated L - 1 times.

    Learnable parameters are stored in the self.params dictionary and will be learned
    using the Solver class.
    """

    def __init__(
        self,
        hidden_dims,
        input_dim=3 * 32 * 32,
        num_classes=10,
        dropout_keep_ratio=1,
        normalization=None,
        reg=0.0,
        weight_scale=1e-2,
        dtype=np.float32,
        seed=None,
    ):
        """Initialize a new FullyConnectedNet.

        Inputs:
        - hidden_dims: A list of integers giving the size of each hidden layer.
        - input_dim: An integer giving the size of the input.
        - num_classes: An integer giving the number of classes to classify.
        - dropout_keep_ratio: Scalar between 0 and 1 giving dropout strength.
            If dropout_keep_ratio=1 then the network should not use dropout at all.
        - normalization: What type of normalization the network should use. Valid values
            are "batchnorm", "layernorm", or None for no normalization (the default).
        - reg: Scalar giving L2 regularization strength.
        - weight_scale: Scalar giving the standard deviation for random
            initialization of the weights.
        - dtype: A numpy datatype object; all computations will be performed using
            this datatype. float32 is faster but less accurate, so you should use
            float64 for numeric gradient checking.
        - seed: If not None, then pass this random seed to the dropout layers.
            This will make the dropout layers deteriminstic so we can gradient check the model.
        """
        self.normalization = normalization
        self.use_dropout = dropout_keep_ratio != 1
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
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        self.params['W1'] = np.random.normal(0.0, weight_scale, (input_dim, hidden_dims[0]))
        self.params['b1'] = np.zeros((1, hidden_dims[0]))
        if self.normalization == 'batchnorm' or self.normalization == 'layernorm' :
          self.params['gamma1'] = np.ones((1, hidden_dims[0]))
          self.params['beta1'] = np.zeros((1, hidden_dims[0]))

        for i in range(2, self.num_layers):
          self.params['W' + str(i)] = np.random.normal(0.0, weight_scale, (hidden_dims[i-2], hidden_dims[i-1]))
          self.params['b' + str(i)] = np.zeros((1, hidden_dims[i-1]))
          if self.normalization == 'batchnorm' or self.normalization == 'layernorm' :
            self.params['gamma' + str(i)] = np.ones((1, hidden_dims[i - 1]))
            self.params['beta' + str(i)] = np.zeros((1, hidden_dims[i - 1]))


        self.params['W' + str(self.num_layers)] = np.random.normal(0.0, weight_scale, (hidden_dims[self.num_layers - 2], num_classes))
        self.params['b' + str(self.num_layers)] = np.zeros((1, num_classes))
        # ????????? ????????? layer?????? batchnorm??? ??????????????? ?????????.
        #if self.normalization == 'batchnorm' :
          #self.params['gamma' + str(self.num_layers)] = np.ones((1, num_classes))
          #self.params['beta' + str(self.num_layers)] = np.zeros((1, num_classes))

        pass

        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        # When using dropout we need to pass a dropout_param dictionary to each
        # dropout layer so that the layer knows the dropout probability and the mode
        # (train / test). You can pass the same dropout_param to each dropout layer.
        self.dropout_param = {}
        if self.use_dropout:
            #print('????????? ?????????')
            self.dropout_param = {"mode": "train", "p": dropout_keep_ratio}
            if seed is not None:
                self.dropout_param["seed"] = seed

        # With batch normalization we need to keep track of running means and
        # variances, so we need to pass a special bn_param object to each batch
        # normalization layer. You should pass self.bn_params[0] to the forward pass
        # of the first batch normalization layer, self.bn_params[1] to the forward
        # pass of the second batch normalization layer, etc.
        self.bn_params = []
        if self.normalization == "batchnorm":
            self.bn_params = [{"mode": "train"} for i in range(self.num_layers - 1)]
        if self.normalization == "layernorm":
            self.bn_params = [{} for i in range(self.num_layers - 1)]

        # Cast all parameters to the correct datatype.
        for k, v in self.params.items():
            self.params[k] = v.astype(dtype)

    def loss(self, X, y=None):
        """Compute loss and gradient for the fully connected net.
        
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
        X = X.astype(self.dtype)
        mode = "test" if y is None else "train"

        # Set train/test mode for batchnorm params and dropout param since they
        # behave differently during training and testing.
        if self.use_dropout:
            self.dropout_param["mode"] = mode
        if self.normalization == "batchnorm":
            for bn_param in self.bn_params:
                bn_param["mode"] = mode
        scores = None
        ############################################################################
        # TODO: Implement the forward pass for the fully connected net, computing  #
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
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        batch_cache_save = []
        layer_cache_save = []
        drop_cache_save = []
        result = {} ## ???????????? ???????????? ?????? dict type ??????
        X_reshape = np.reshape(X,(X.shape[0],np.prod(np.shape(X)[1:])))
        ## ?????????????????? ????????? ???????????? forward prop?????????
        result['z1'], cache = affine_forward(X_reshape, self.params['W1'], self.params['b1'])
        ## batchnorm??? ?????????????????? activation func ???????????? ?????? batchnorm??? ???????????????.
        if self.normalization == 'batchnorm' :
          result['z1'], batch_cache = batchnorm_forward(result['z1'], self.params['gamma1'], self.params['beta1'], self.bn_params[0])
          batch_cache_save.append(batch_cache)
        if self.normalization == 'layernorm' :
          result['z1'], layer_cache = layernorm_forward(result['z1'], self.params['gamma1'], self.params['beta1'], self.bn_params[0])
          layer_cache_save.append(layer_cache)
        ## activation func ??????
        result['a1'], cache = relu_forward(result['z1'])
        ## activ ?????? ??? dropout ??????
        if self.use_dropout == True :
          result['a1'], drop_cache = dropout_forward(result['a1'], self.dropout_param)
          drop_cache_save.append(drop_cache)

        for i in range(2, self.num_layers): ## ?????? hidden layer?????? ??????????????? ???????????????.
          result['z' + str(i)], cache = affine_forward(result['a' + str(i-1)], self.params['W' + str(i)], self.params['b' + str(i)])
          #output layer????????? batchnorm??? activation?????? ??????????????????.
          if self.normalization == 'batchnorm' :
            result['z' + str(i)], batch_cache = batchnorm_forward(result['z' + str(i)], self.params['gamma' + str(i)], self.params['beta' + str(i)], self.bn_params[i-1])
            batch_cache_save.append(batch_cache)
          elif self.normalization == 'layernorm' :
            result['z' + str(i)], layer_cache = layernorm_forward(result['z' + str(i)], self.params['gamma' + str(i)], self.params['beta' + str(i)], self.bn_params[i-1])
            layer_cache_save.append(layer_cache)          
          ## activ func ?????? 
          result['a' + str(i)], cache = relu_forward(result['z' + str(i)])
          ## dropout ??????
          if self.use_dropout == True :
            result['a' + str(i)], drop_cache = dropout_forward(result['a' + str(i)], self.dropout_param)
            drop_cache_save.append(drop_cache)
        ## ????????? ????????? ???????????? ??? ??? ???????????? forwar prop?????????.
        result['z' + str(self.num_layers)], cache = affine_forward(result['a' + str(self.num_layers - 1)], self.params['W' + str(self.num_layers)], self.params['b' + str(self.num_layers)])
        ## ????????? ????????? ???????????? dropout??? ??????
        #if self.use_dropout == True :
          #result['z' + str(self.num_layers)], drop_cache = dropout_forward(result['z' + str(self.num_layers)], self.dropout_param)
          #drop_cache_save.append(drop_cache)
        scores = result['z' + str(self.num_layers)] ## ?????? ???????????? ?????? test?????? scores??? ???????????????.

        
        

        pass

        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        # If test mode return early.
        if mode == "test":
            return scores

        loss, grads = 0.0, {}
        ############################################################################
        # TODO: Implement the backward pass for the fully connected net. Store the #
        # loss in the loss variable and gradients in the grads dictionary. Compute #
        # data loss using softmax, and make sure that grads[k] holds the gradients #
        # for self.params[k]. Don't forget to add L2 regularization!               #
        #                                                                          #
        # When using batch/layer normalization, you don't need to regularize the   #
        # scale and shift parameters.                                              #
        #                                                                          #
        # NOTE: To ensure that your implementation matches ours and you pass the   #
        # automated tests, make sure that your L2 regularization includes a factor #
        # of 0.5 to simplify the expression for the gradient.                      #
        ############################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        #print(batch_cache_save)
        loss, dx = softmax_loss(result['z' + str(self.num_layers)], y)
        L2_reguralize = 0 ## L2 reg??? 0?????? ?????????
        for i in range(1, self.num_layers + 1): ## L2 reg?????? ??????????????? ??????
          L2_reguralize += np.sum(self.params['W' + str(i)] ** 2)
        L2_reguralize *= self.reg / 2 ## leg strength ??????

        loss = loss + L2_reguralize
        #print(batch_cache_save[1])

        result['dz' + str(self.num_layers)] = dx ## softmax??? derivative?????? result?????? ??????????????????.
        ## outputlayer dropout??? ??????
        #if self.use_dropout == True :
          #result['dz' + str(self.num_layers)] = dropout_backward(result['dz' + str(self.num_layers)], drop_cache_save[self.num_layers-1])
        for i in range(self.num_layers, 1, -1):
          grads['W' + str(i)] = np.dot(result['a' + str(i-1)].T, result['dz' + str(i)]) + (self.reg * self.params['W' + str(i)])
          grads['b' + str(i)] = np.sum(result['dz' + str(i)], axis = 0, keepdims = True)
          result['da' + str(i-1)] = np.dot(result['dz' + str(i)], self.params['W' + str(i)].T) # ?????? ????????? prev layer??? grads??? ????????? ?????? ?????? ???????????? derivative?????? ?????????
          ## active backprop ?????? drop backprop??? ????????????
          if self.use_dropout == True :
            result['da' + str(i-1)] = dropout_backward(result['da' + str(i-1)], drop_cache_save[i-2])
          result['dz' + str(i-1)] = np.where(result['z' + str(i-1)] <= 0, 0, 1) * result['da' + str(i-1)] #?????? ????????? prev layer??? relu_back prop??? ??????
          #print('????????? ',i,np.shape(result['dz' + str(i-1)]))
          if self.normalization == 'batchnorm': # ????????? ???????????? ????????? batchnorm??? ????????? batchnorm backprop??? ??????.
            result['dz' + str(i-1)], grads['gamma' + str(i-1)], grads['beta' + str(i-1)]  = batchnorm_backward(result['dz' + str(i-1)], batch_cache_save[i-2])
          elif self.normalization == 'layernorm' :
            result['dz' + str(i-1)], grads['gamma' + str(i-1)], grads['beta' + str(i-1)]  = layernorm_backward(result['dz' + str(i-1)], layer_cache_save[i-2])
        grads['W1'] = np.dot(X_reshape.T, result['dz1']) + (self.reg * self.params['W1'])
        grads['b1'] = np.sum(result['dz1'], axis = 0, keepdims = True)

        pass

        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        return loss, grads
