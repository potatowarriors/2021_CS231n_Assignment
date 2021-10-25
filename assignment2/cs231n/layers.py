from builtins import range
import numpy as np


def affine_forward(x, w, b):
    """Computes the forward pass for an affine (fully connected) layer.

    The input x has shape (N, d_1, ..., d_k) and contains a minibatch of N
    examples, where each example x[i] has shape (d_1, ..., d_k). We will
    reshape each input into a vector of dimension D = d_1 * ... * d_k, and
    then transform it to an output vector of dimension M.

    Inputs:
    - x: A numpy array containing input data, of shape (N, d_1, ..., d_k)
    - w: A numpy array of weights, of shape (D, M)
    - b: A numpy array of biases, of shape (M,)

    Returns a tuple of:
    - out: output, of shape (N, M)
    - cache: (x, w, b)
    """
    out = None
    ###########################################################################
    # TODO: Copy over your solution from Assignment 1.                        #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    t = np.reshape(x,(x.shape[0],np.prod(np.shape(x)[1:])))
    out = np.dot(t,w) + b
    #print('affine', np.shape(out))

    pass

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    cache = (x, w, b)
    return out, cache


def affine_backward(dout, cache):
    """Computes the backward pass for an affine (fully connected) layer.

    Inputs:
    - dout: Upstream derivative, of shape (N, M)
    - cache: Tuple of:
      - x: Input data, of shape (N, d_1, ... d_k)
      - w: Weights, of shape (D, M)
      - b: Biases, of shape (M,)

    Returns a tuple of:
    - dx: Gradient with respect to x, of shape (N, d1, ..., d_k)
    - dw: Gradient with respect to w, of shape (D, M)
    - db: Gradient with respect to b, of shape (M,)
    """
    x, w, b = cache
    dx, dw, db = None, None, None
    ###########################################################################
    # TODO: Copy over your solution from Assignment 1.                        #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    dim_shape = np.prod(x[0].shape)
    N = x.shape[0]
    X = x.reshape(N, dim_shape)
    # input gradient
    dx = dout.dot(w.T)
    dx = dx.reshape(x.shape)
    # weight gradient
    dw = X.T.dot(dout)
    # bias gradient
    db = dout.sum(axis=0)
    pass

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return dx, dw, db


def relu_forward(x):
    """Computes the forward pass for a layer of rectified linear units (ReLUs).

    Input:
    - x: Inputs, of any shape

    Returns a tuple of:
    - out: Output, of the same shape as x
    - cache: x
    """
    out = None
    ###########################################################################
    # TODO: Copy over your solution from Assignment 1.                        #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    #out = np.where(x<=0, 0, x)
    out = np.maximum(0,x)
    


    pass

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    cache = x
    return out, cache


def relu_backward(dout, cache):
    """Computes the backward pass for a layer of rectified linear units (ReLUs).

    Input:
    - dout: Upstream derivatives, of any shape
    - cache: Input x, of same shape as dout

    Returns:
    - dx: Gradient with respect to x
    """
    dx, x = None, cache
    ###########################################################################
    # TODO: Copy over your solution from Assignment 1.                        #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    dx = dout * (x > 0)

    pass

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return dx


def softmax_loss(x, y):
    """Computes the loss and gradient for softmax classification.

    Inputs:
    - x: Input data, of shape (N, C) where x[i, j] is the score for the jth
      class for the ith input.
    - y: Vector of labels, of shape (N,) where y[i] is the label for x[i] and
      0 <= y[i] < C

    Returns a tuple of:
    - loss: Scalar giving the loss
    - dx: Gradient of the loss with respect to x
    """
    loss, dx = None, None

    ###########################################################################
    # TODO: Copy over your solution from Assignment 1.                        #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    '''shifted_logits = x - np.max(x, axis=1, keepdims=True)
    Z = np.sum(np.exp(shifted_logits), axis=1, keepdims=True)
    log_probs = shifted_logits - np.log(Z)
    probs = np.exp(log_probs)
    N = x.shape[0]
    loss = -np.sum(log_probs[np.arange(N), y]) / N
    dx = probs.copy()
    dx[np.arange(N), y] -= 1
    dx /= N'''
    
    num_train = x.shape[0]

    x = np.exp(x)
    temp_sum = np.sum(x, axis = 1, keepdims = True)
    x = x / temp_sum
    softmax_result = x
    trans_y = np.zeros((x.shape[0],x.shape[1]))
    trans_y[np.arange(x.shape[0]), y] += 1
    x = - np.log(x)
    x = x * trans_y
    x_sum = np.sum(x)
    loss = x_sum / num_train
    dx = softmax_result - trans_y
    dx = dx / num_train
    

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return loss, dx


def batchnorm_forward(x, gamma, beta, bn_param):
    """Forward pass for batch normalization.

    During training the sample mean and (uncorrected) sample variance are
    computed from minibatch statistics and used to normalize the incoming data.
    During training we also keep an exponentially decaying running mean of the
    mean and variance of each feature, and these averages are used to normalize
    data at test-time.

    At each timestep we update the running averages for mean and variance using
    an exponential decay based on the momentum parameter:

    running_mean = momentum * running_mean + (1 - momentum) * sample_mean
    running_var = momentum * running_var + (1 - momentum) * sample_var

    Note that the batch normalization paper suggests a different test-time
    behavior: they compute sample mean and variance for each feature using a
    large number of training images rather than using a running average. For
    this implementation we have chosen to use running averages instead since
    they do not require an additional estimation step; the torch7
    implementation of batch normalization also uses running averages.

    Input:
    - x: Data of shape (N, D)
    - gamma: Scale parameter of shape (D,)
    - beta: Shift paremeter of shape (D,)
    - bn_param: Dictionary with the following keys:
      - mode: 'train' or 'test'; required
      - eps: Constant for numeric stability
      - momentum: Constant for running mean / variance.
      - running_mean: Array of shape (D,) giving running mean of features
      - running_var Array of shape (D,) giving running variance of features

    Returns a tuple of:
    - out: of shape (N, D)
    - cache: A tuple of values needed in the backward pass
    """
    mode = bn_param["mode"]
    eps = bn_param.get("eps", 1e-5)
    momentum = bn_param.get("momentum", 0.9)

    N, D = x.shape
    running_mean = bn_param.get("running_mean", np.zeros(D, dtype=x.dtype))
    running_var = bn_param.get("running_var", np.zeros(D, dtype=x.dtype))

    out, cache = None, None
    if mode == "train":
        #######################################################################
        # TODO: Implement the training-time forward pass for batch norm.      #
        # Use minibatch statistics to compute the mean and variance, use      #
        # these statistics to normalize the incoming data, and scale and      #
        # shift the normalized data using gamma and beta.                     #
        #                                                                     #
        # You should store the output in the variable out. Any intermediates  #
        # that you need for the backward pass should be stored in the cache   #
        # variable.                                                           #
        #                                                                     #
        # You should also use your computed sample mean and variance together #
        # with the momentum variable to update the running mean and running   #
        # variance, storing your result in the running_mean and running_var   #
        # variables.                                                          #
        #                                                                     #
        # Note that though you should be keeping track of the running         #
        # variance, you should normalize the data based on the standard       #
        # deviation (square root of variance) instead!                        #
        # Referencing the original paper (https://arxiv.org/abs/1502.03167)   #
        # might prove to be helpful.                                          #
        #######################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        #print(running_mean)
        sample_mean = np.sum(x, axis = 0) / N # batch sample의 평균을 구한다
        sample_var = np.sum((x - sample_mean) ** 2, axis = 0) / N # batch smaple의 분산을 구한다
        x_hat = (x - sample_mean) / np.sqrt(sample_var + eps) # bach norm을 적용시킨 x_hat을 구한다.
        out = gamma * x_hat + beta # 학습가능한 파라미터를 적용시킨 out을 반환해준다.
        running_mean = momentum * running_mean + (1 - momentum) * sample_mean # 지수적 가중 평균을 이용해 running mean을 업데이트 시켜준다
        #print('running',np.shape(running_mean))
        running_var = momentum * running_var + (1 - momentum) * sample_var # 지수적 가중 편균을 이용해 running variance를 업데이트 시켜준다.
        #print('runing',np.shape(running_var))
        cache = {'x' : x, 'x_hat' : x_hat, 'gamma':gamma, 'sample_mean':sample_mean, 'sample_var':sample_var, 'eps' : eps}
        #cache = (x, x_hat, gamma, sample_mean, sample_var, eps)


        
        

        pass

        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        #######################################################################
        #                           END OF YOUR CODE                          #
        #######################################################################
    elif mode == "test":
        #######################################################################
        # TODO: Implement the test-time forward pass for batch normalization. #
        # Use the running mean and variance to normalize the incoming data,   #
        # then scale and shift the normalized data using gamma and beta.      #
        # Store the result in the out variable.                               #
        #######################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****    
        x_hat = (x - running_mean) / np.sqrt(running_var + eps) # test시에는 지수적 가중 평균으로 업데이트한 ruuning mean, variance를 이용해 x_hat을 구한다
        out = gamma * x_hat + beta # 학습가능한 파라미터를 적용시킨 out을 반환시켜준다.
        pass

        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        #######################################################################
        #                          END OF YOUR CODE                           #
        #######################################################################
    else:
        raise ValueError('Invalid forward batchnorm mode "%s"' % mode)

    # Store the updated running means back into bn_param
    bn_param["running_mean"] = running_mean
    bn_param["running_var"] = running_var

    return out, cache


def batchnorm_backward(dout, cache):
    """Backward pass for batch normalization.

    For this implementation, you should write out a computation graph for
    batch normalization on paper and propagate gradients backward through
    intermediate nodes.

    Inputs:
    - dout: Upstream derivatives, of shape (N, D)
    - cache: Variable of intermediates from batchnorm_forward.

    Returns a tuple of:
    - dx: Gradient with respect to inputs x, of shape (N, D)
    - dgamma: Gradient with respect to scale parameter gamma, of shape (D,)
    - dbeta: Gradient with respect to shift parameter beta, of shape (D,)
    """
    dx, dgamma, dbeta = None, None, None
    ###########################################################################
    # TODO: Implement the backward pass for batch normalization. Store the    #
    # results in the dx, dgamma, and dbeta variables.                         #
    # Referencing the original paper (https://arxiv.org/abs/1502.03167)       #
    # might prove to be helpful.                                              #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    x = cache['x']
    x_hat = cache['x_hat']
    gamma = cache['gamma']
    batch_mean = cache['sample_mean']
    batch_var = cache['sample_var']
    eps = cache['eps']
    #x, x_hat, gamma, batch_mean, batch_var, eps = cache #캐쉬로 필요한 param 불러온다
    N, D = np.shape(dout) 
    dx_hat = dout * gamma
    dbatch_var = np.sum(dx_hat * (x - batch_mean) * ((-1/2) * ((batch_var + eps) ** (-3/2))), axis = 0)
    dbatch_mean = np.sum(dx_hat * (-1 / np.sqrt(batch_var + eps)), axis = 0) + (dbatch_var * (np.sum(-2 * (x - batch_mean), axis = 0) / N))
    dx = dx_hat * (1 / np.sqrt(batch_var + eps)) + (dbatch_var * ((2 * (x - batch_mean)) / N)) + (dbatch_mean / N)
    dgamma = np.sum(dout * x_hat, axis = 0)
    dbeta = np.sum(dout, axis = 0)

    pass

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################

    return dx, dgamma, dbeta


def batchnorm_backward_alt(dout, cache):
    """Alternative backward pass for batch normalization.

    For this implementation you should work out the derivatives for the batch
    normalizaton backward pass on paper and simplify as much as possible. You
    should be able to derive a simple expression for the backward pass.
    See the jupyter notebook for more hints.

    Note: This implementation should expect to receive the same cache variable
    as batchnorm_backward, but might not use all of the values in the cache.

    Inputs / outputs: Same as batchnorm_backward
    """
    dx, dgamma, dbeta = None, None, None
    ###########################################################################
    # TODO: Implement the backward pass for batch normalization. Store the    #
    # results in the dx, dgamma, and dbeta variables.                         #
    #                                                                         #
    # After computing the gradient with respect to the centered inputs, you   #
    # should be able to compute gradients with respect to the inputs in a     #
    # single statement; our implementation fits on a single 80-character line.#
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    x = cache['x']
    x_hat = cache['x_hat']
    gamma = cache['gamma']
    batch_mean = cache['sample_mean']
    batch_var = cache['sample_var']
    eps = cache['eps']
    N, D = np.shape(dout) 
    dx_hat = dout * gamma
    dbatch_var = np.sum(dx_hat * (x - batch_mean) * ((-1/2) * ((batch_var + eps) ** (-3/2))), axis = 0)
    dbatch_mean = np.sum(dx_hat * (-1 / np.sqrt(batch_var + eps)), axis = 0) + (dbatch_var * (np.sum(-2 * (x - batch_mean), axis = 0) / N))
    dx = dx_hat * (1 / np.sqrt(batch_var + eps)) + (dbatch_var * ((2 * (x - batch_mean)) / N)) + (dbatch_mean / N)
    dgamma = np.sum(dout * x_hat, axis = 0)
    dbeta = np.sum(dout, axis = 0)    

    pass

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################

    return dx, dgamma, dbeta


def layernorm_forward(x, gamma, beta, ln_param):
    """Forward pass for layer normalization.

    During both training and test-time, the incoming data is normalized per data-point,
    before being scaled by gamma and beta parameters identical to that of batch normalization.

    Note that in contrast to batch normalization, the behavior during train and test-time for
    layer normalization are identical, and we do not need to keep track of running averages
    of any sort.

    Input:
    - x: Data of shape (N, D)
    - gamma: Scale parameter of shape (D,)
    - beta: Shift paremeter of shape (D,)
    - ln_param: Dictionary with the following keys:
        - eps: Constant for numeric stability

    Returns a tuple of:
    - out: of shape (N, D)
    - cache: A tuple of values needed in the backward pass
    """
    out, cache = None, None
    eps = ln_param.get("eps", 1e-5)
    ###########################################################################
    # TODO: Implement the training-time forward pass for layer norm.          #
    # Normalize the incoming data, and scale and  shift the normalized data   #
    #  using gamma and beta.                                                  #
    # HINT: this can be done by slightly modifying your training-time         #
    # implementation of  batch normalization, and inserting a line or two of  #
    # well-placed code. In particular, can you think of any matrix            #
    # transformations you could perform, that would enable you to copy over   #
    # the batch norm code and leave it almost unchanged?                      #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    # batchnorm에서 적절하게 transepose시켜준다. batch는 batchsample기준으로 normalize하고 layernorm은 feature dim 기준으로 normalize한다
    x = x.T
    N,D = np.shape(x)
    sample_mean = np.sum(x, axis = 0) / N
    sample_var = np.sum((x - sample_mean) ** 2, axis = 0) / N 
    x_hat = (x - sample_mean) / np.sqrt(sample_var + eps)
    x_hat = x_hat.T
    out = gamma * x_hat + beta 
    cache = {'x' : x, 'x_hat' : x_hat, 'gamma':gamma, 'sample_mean':sample_mean, 'sample_var':sample_var, 'eps' : eps}
    pass

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return out, cache


def layernorm_backward(dout, cache):
    """Backward pass for layer normalization.

    For this implementation, you can heavily rely on the work you've done already
    for batch normalization.

    Inputs:
    - dout: Upstream derivatives, of shape (N, D)
    - cache: Variable of intermediates from layernorm_forward.

    Returns a tuple of:
    - dx: Gradient with respect to inputs x, of shape (N, D)
    - dgamma: Gradient with respect to scale parameter gamma, of shape (D,)
    - dbeta: Gradient with respect to shift parameter beta, of shape (D,)
    """
    dx, dgamma, dbeta = None, None, None
    ###########################################################################
    # TODO: Implement the backward pass for layer norm.                       #
    #                                                                         #
    # HINT: this can be done by slightly modifying your training-time         #
    # implementation of batch normalization. The hints to the forward pass    #
    # still apply!                                                            #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    #dx, dgamma, dbeta = batchnorm_backward_alt(dout.T, cache)
    # transpose gradients w.r.t. input, x, to their original dims
    x = cache['x']
    x_hat = cache['x_hat']
    gamma = cache['gamma']
    batch_mean = cache['sample_mean']
    batch_var = cache['sample_var']
    eps = cache['eps']
    #x, x_hat, gamma, batch_mean, batch_var, eps = cache #캐쉬로 필요한 param 불러온다
    # foward 처럼 적당히 필요한 놈들만 transpose 시켜줬다 풀어준다.
    dout = dout.T
    N, D = np.shape(dout) 
    dx_hat = dout.T * gamma
    dx_hat = dx_hat.T
    dbatch_var = np.sum(dx_hat * (x - batch_mean) * ((-1/2) * ((batch_var + eps) ** (-3/2))), axis = 0)
    dbatch_mean = np.sum(dx_hat * (-1 / np.sqrt(batch_var + eps)), axis = 0) + (dbatch_var * (np.sum(-2 * (x - batch_mean), axis = 0) / N))
    dx = dx_hat * (1 / np.sqrt(batch_var + eps)) + (dbatch_var * ((2 * (x - batch_mean)) / N)) + (dbatch_mean / N)
    dx = dx.T
    dgamma = np.sum(dout.T * x_hat, axis = 0)
    dbeta = np.sum(dout.T, axis = 0)

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return dx, dgamma, dbeta


def dropout_forward(x, dropout_param):
    """Forward pass for inverted dropout.

    Note that this is different from the vanilla version of dropout.
    Here, p is the probability of keeping a neuron output, as opposed to
    the probability of dropping a neuron output.
    See http://cs231n.github.io/neural-networks-2/#reg for more details.

    Inputs:
    - x: Input data, of any shape
    - dropout_param: A dictionary with the following keys:
      - p: Dropout parameter. We keep each neuron output with probability p.
      - mode: 'test' or 'train'. If the mode is train, then perform dropout;
        if the mode is test, then just return the input.
      - seed: Seed for the random number generator. Passing seed makes this
        function deterministic, which is needed for gradient checking but not
        in real networks.

    Outputs:
    - out: Array of the same shape as x.
    - cache: tuple (dropout_param, mask). In training mode, mask is the dropout
      mask that was used to multiply the input; in test mode, mask is None.
    """
    p, mode = dropout_param["p"], dropout_param["mode"]
    if "seed" in dropout_param:
        np.random.seed(dropout_param["seed"])

    mask = None
    out = None

    if mode == "train":
        #######################################################################
        # TODO: Implement training phase forward pass for inverted dropout.   #
        # Store the dropout mask in the mask variable.                        #
        #######################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        N,D = np.shape(x)
        mask = (np.random.rand(N,D) < p) / p
        out = x * mask

        pass

        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        #######################################################################
        #                           END OF YOUR CODE                          #
        #######################################################################
    elif mode == "test":
        #######################################################################
        # TODO: Implement the test phase forward pass for inverted dropout.   #
        #######################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        out = x

        pass

        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        #######################################################################
        #                            END OF YOUR CODE                         #
        #######################################################################

    cache = (dropout_param, mask)
    out = out.astype(x.dtype, copy=False)

    return out, cache


def dropout_backward(dout, cache):
    """Backward pass for inverted dropout.

    Inputs:
    - dout: Upstream derivatives, of any shape
    - cache: (dropout_param, mask) from dropout_forward.
    """
    dropout_param, mask = cache
    mode = dropout_param["mode"]

    dx = None
    if mode == "train":
        #######################################################################
        # TODO: Implement training phase backward pass for inverted dropout   #
        #######################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        dx = dout * mask

        pass

        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        #######################################################################
        #                          END OF YOUR CODE                           #
        #######################################################################
    elif mode == "test":
        dx = dout
    return dx


def conv_forward_naive(x, w, b, conv_param):
    """A naive implementation of the forward pass for a convolutional layer.

    The input consists of N data points, each with C channels, height H and
    width W. We convolve each input with F different filters, where each filter
    spans all C channels and has height HH and width WW.

    Input:
    - x: Input data of shape (N, C, H, W)
    - w: Filter weights of shape (F, C, HH, WW)
    - b: Biases, of shape (F,)
    - conv_param: A dictionary with the following keys:
      - 'stride': The number of pixels between adjacent receptive fields in the
        horizontal and vertical directions.
      - 'pad': The number of pixels that will be used to zero-pad the input.

    During padding, 'pad' zeros should be placed symmetrically (i.e equally on both sides)
    along the height and width axes of the input. Be careful not to modfiy the original
    input x directly.

    Returns a tuple of:
    - out: Output data, of shape (N, F, H', W') where H' and W' are given by
      H' = 1 + (H + 2 * pad - HH) / stride
      W' = 1 + (W + 2 * pad - WW) / stride
    - cache: (x, w, b, conv_param)
    """
    out = None
    ###########################################################################
    # TODO: Implement the convolutional forward pass.                         #
    # Hint: you can use the function np.pad for padding.                      #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    stride = conv_param['stride']
    pad = conv_param['pad']

    N, C, H, W = np.shape(x)
    F, __, FH, FW = np.shape(w)

    x_pad = np.pad(x, ((0,0), (0,0), (pad,pad), (pad, pad)))
    H_pad, W_pad = x_pad.shape[2], x_pad.shape[3]

    out_W = int(1 + (W - FW + 2 * pad) / stride) # input과 filter를 고려해 가로로 몇번 슬라이드 해야하는지 구한다
    #print(num_slide_width)
    out_H = int(1 + (H - FH + 2 * pad) / stride) # input과 filter를 고려해 세로로 몇번 슬라이드 해야하는지 구한다
    #print(num_slide_hight)
    num_filter_feature = C * FH * FW
    #결과값을 zeros로 초기화 시켜준다
    out = np.zeros((N, F, out_H, out_W))

    # row방향으로는 filter의 dimension 만큼 값들을 쌓아줄거니까 지정하고 colum방향으로는 슬라이드 총 횟수만큼 나올거니까 곱해서 넣어준다
    x_col = np.zeros((num_filter_feature, out_H * out_W)) # 이넘은 2dimension matrix다 각 colum마다 filter랑 결합할 놈들을 input에서 뽑아와서 쌓아준거다

    # filter 수 만큼의 row에 filterfeature개 만큼 컬럼을 싸아준다. 
    w_row = w.reshape(F, num_filter_feature)
    # broadcast 에러 안나게 1로 reshape
    #b = b.reshape(F,1)


    for n in range(N) : # input image수 N개에 대한 반복
      total_slide_num = 0
      for i in range(0, H_pad - FH + 1, stride) : #y방향으로 슬라이드 시켜준다
        #print('y넘어갔냐')
        for j in range(0, W_pad - FW + 1, stride) : #x방향으로 슬라이드 시켜준다
          #print('언제 에러나는거야 대체', total_slide_num)
          x_col[:, total_slide_num] = x_pad[n, :, i:i + FH, j:j+FW].reshape(num_filter_feature)
          ## traingset수는 n index로진행시켜준다 channel은 전체 채널을 사용한다. x_pad에서 x,y축 방향으로 시작점으로부터 filter크기만큼을 가져온다 그걸 최종적으로 filterfeature수로 reshape한다
          total_slide_num += 1
      #print(np.shape(np.dot(w_row,x_col) + b))
      #print(np.shape(out[1]))
      out[n] = (np.dot(w_row, x_col) + b.reshape(-1,1)).reshape(F, out_H, out_W)
    
    x = x_pad


    
    


    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    cache = (x, w, b, conv_param)
    return out, cache


def conv_backward_naive(dout, cache):
    """A naive implementation of the backward pass for a convolutional layer.

    Inputs:
    - dout: Upstream derivatives.
    - cache: A tuple of (x, w, b, conv_param) as in conv_forward_naive

    Returns a tuple of:
    - dx: Gradient with respect to x
    - dw: Gradient with respect to w
    - db: Gradient with respect to b
    """
    dx, dw, db = None, None, None
    ###########################################################################
    # TODO: Implement the convolutional backward pass.                        #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    x_pad, w, b, conv_param = cache
    N, F, outH, outW = dout.shape
    N, C, Hpad, Wpad = x_pad.shape
    FH, FW = w.shape[2], w.shape[3]
    stride = conv_param['stride']
    pad = conv_param['pad']
    #print(np.shape(dout))

    num_filter_feature = C * FW * FH
    #print(num_filter_feature)
    w_row = w.reshape(F,num_filter_feature)
    #print(np.shape(w_row))
    dx = np.zeros((N, C, Hpad - 2*pad, Wpad - 2*pad))
    dw = np.zeros(w.shape)
    db = np.zeros(b.shape)

    x_col = np.zeros((num_filter_feature, outH*outW))

    for n in range(N) :
      z = dout[n].reshape(F, outH * outW)      ## z = [F, outshape]
      dz = np.dot(w_row.T, z)        ##dz = [num_filter_feature, outshape]
      total_slide_num = 0
      dx_temp = np.zeros((C, Hpad, Wpad))       ## 패딩 사이즈로 임시 초기화
      for i in range(0, Hpad - FH + 1 , stride):
        for j in range(0, Wpad - FW + 1 ,stride) :
          dx_temp[:, i:i+FH, j:j+FW] += dz[:,total_slide_num].reshape(C, FH, FW)     ## num_filter_fetrue를 디멘젼에 맞게 풀어서 업데이트
          x_col[:,total_slide_num] = x_pad[n, :, i:i+FH, j:j+FW].reshape(num_filter_feature)    ## x_col을 만들어준다
          total_slide_num += 1
      dx[n] = dx_temp[:, pad:-pad, pad:-pad] ## 패딩 부분을 자른다
      dw += np.dot(z, x_col.T).reshape(F, C, FH, FW) ## dw를 구한 후 디멘젼에 맞춰 reshape해준다 n개 만큼 계속 업데이트 된다.
      db += np.sum(z, axis = 1)



    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return dx, dw, db


def max_pool_forward_naive(x, pool_param):
    """A naive implementation of the forward pass for a max-pooling layer.

    Inputs:
    - x: Input data, of shape (N, C, H, W)
    - pool_param: dictionary with the following keys:
      - 'pool_height': The height of each pooling region
      - 'pool_width': The width of each pooling region
      - 'stride': The distance between adjacent pooling regions

    No padding is necessary here, eg you can assume:
      - (H - pool_height) % stride == 0
      - (W - pool_width) % stride == 0

    Returns a tuple of:
    - out: Output data, of shape (N, C, H', W') where H' and W' are given by
      H' = 1 + (H - pool_height) / stride
      W' = 1 + (W - pool_width) / stride
    - cache: (x, pool_param)
    """
    out = None
    ###########################################################################
    # TODO: Implement the max-pooling forward pass                            #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    N, C, H, W = x.shape
    stride = pool_param['stride']
    PH = pool_param['pool_height']
    PW = pool_param['pool_width']
    outH = int(1 + (H - PH) / stride)
    outW = int(1 + (W - PW) / stride)


    out = np.zeros((N, C, outH, outW)) # output 초기화
    for index in range(N):
        out_col = np.zeros((C, outH*outW))
        neuron = 0
        for i in range(0, H - PH + 1, stride):
            for j in range(0, W - PW + 1, stride):
                pool_region = x[index,:,i:i+PH,j:j+PW].reshape(C,PH*PW) ## pool_region에 maxpolling 할 값들을 넣는다
                out_col[:,neuron] = pool_region.max(axis=1) #pool_region중 제일 큰놈을 out_col에 할당시킨다.
                neuron += 1
        out[index] = out_col.reshape(C, outH, outW) # out put size에 맞게 reshape시켜준다
    pass

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    cache = (x, pool_param)
    return out, cache


def max_pool_backward_naive(dout, cache):
    """A naive implementation of the backward pass for a max-pooling layer.

    Inputs:
    - dout: Upstream derivatives
    - cache: A tuple of (x, pool_param) as in the forward pass.

    Returns:
    - dx: Gradient with respect to x
    """
    dx = None
    ###########################################################################
    # TODO: Implement the max-pooling backward pass                           #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    x, pool_param = cache
    N, C, outH, outW = dout.shape
    H, W = x.shape[2], x.shape[3]
    stride = pool_param['stride']
    PH, PW = pool_param['pool_height'], pool_param['pool_width']


    dx = np.zeros(x.shape)
    ## 차근차근 forward를 반대로 한다
    for index in range(N):
        dout_row = dout[index].reshape(C, outH*outW)
        total_slide_num = 0
        for i in range(0, H-PH+1, stride):
            for j in range(0, W-PW+1, stride):
                pool_region = x[index,:,i:i+PH,j:j+PW].reshape(C,PH*PW)
                max_pool_indices = pool_region.argmax(axis=1)
                dout_cur = dout_row[:,total_slide_num]
                total_slide_num += 1
  
                dmax_pool = np.zeros(pool_region.shape)
                dmax_pool[np.arange(C),max_pool_indices] = dout_cur
                dx[index,:,i:i+PH,j:j+PW] += dmax_pool.reshape(C,PH,PW)
    pass

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return dx


def spatial_batchnorm_forward(x, gamma, beta, bn_param):
    """Computes the forward pass for spatial batch normalization.

    Inputs:
    - x: Input data of shape (N, C, H, W)
    - gamma: Scale parameter, of shape (C,)
    - beta: Shift parameter, of shape (C,)
    - bn_param: Dictionary with the following keys:
      - mode: 'train' or 'test'; required
      - eps: Constant for numeric stability
      - momentum: Constant for running mean / variance. momentum=0 means that
        old information is discarded completely at every time step, while
        momentum=1 means that new information is never incorporated. The
        default of momentum=0.9 should work well in most situations.
      - running_mean: Array of shape (D,) giving running mean of features
      - running_var Array of shape (D,) giving running variance of features

    Returns a tuple of:
    - out: Output data, of shape (N, C, H, W)
    - cache: Values needed for the backward pass
    """
    out, cache = None, None

    ###########################################################################
    # TODO: Implement the forward pass for spatial batch normalization.       #
    #                                                                         #
    # HINT: You can implement spatial batch normalization by calling the      #
    # vanilla version of batch normalization you implemented above.           #
    # Your implementation should be very short; ours is less than five lines. #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    N, C, H, W = x.shape
    x = x.transpose(0,2,3,1).reshape(N*H*W, C)
    out, cache = batchnorm_forward(x, gamma, beta, bn_param)
    out = out.reshape(N, H, W, C).transpose(0,3,1,2)

    pass

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################

    return out, cache


def spatial_batchnorm_backward(dout, cache):
    """Computes the backward pass for spatial batch normalization.

    Inputs:
    - dout: Upstream derivatives, of shape (N, C, H, W)
    - cache: Values from the forward pass

    Returns a tuple of:
    - dx: Gradient with respect to inputs, of shape (N, C, H, W)
    - dgamma: Gradient with respect to scale parameter, of shape (C,)
    - dbeta: Gradient with respect to shift parameter, of shape (C,)
    """
    dx, dgamma, dbeta = None, None, None

    ###########################################################################
    # TODO: Implement the backward pass for spatial batch normalization.      #
    #                                                                         #
    # HINT: You can implement spatial batch normalization by calling the      #
    # vanilla version of batch normalization you implemented above.           #
    # Your implementation should be very short; ours is less than five lines. #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    N, C, H, W = dout.shape
    dout = dout.transpose(0,2,3,1).reshape(N*H*W, C)
    dx, dgamma, dbeta = batchnorm_backward(dout, cache)
    dx = dx.reshape(N, H, W, C).transpose(0,3,1,2)

    pass

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################

    return dx, dgamma, dbeta


def spatial_groupnorm_forward(x, gamma, beta, G, gn_param):
    """Computes the forward pass for spatial group normalization.
    
    In contrast to layer normalization, group normalization splits each entry in the data into G
    contiguous pieces, which it then normalizes independently. Per-feature shifting and scaling
    are then applied to the data, in a manner identical to that of batch normalization and layer
    normalization.

    Inputs:
    - x: Input data of shape (N, C, H, W)
    - gamma: Scale parameter, of shape (1, C, 1, 1)
    - beta: Shift parameter, of shape (1, C, 1, 1)
    - G: Integer mumber of groups to split into, should be a divisor of C
    - gn_param: Dictionary with the following keys:
      - eps: Constant for numeric stability

    Returns a tuple of:
    - out: Output data, of shape (N, C, H, W)
    - cache: Values needed for the backward pass
    """
    out, cache = None, None
    eps = gn_param.get("eps", 1e-5)
    ###########################################################################
    # TODO: Implement the forward pass for spatial group normalization.       #
    # This will be extremely similar to the layer norm implementation.        #
    # In particular, think about how you could transform the matrix so that   #
    # the bulk of the code is similar to both train-time batch normalization  #
    # and layer normalization!                                                #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    N, C, H, W = x.shape
    size = (N*G, C//G *H*W)
    x = x.reshape(size).T
    gamma = gamma.reshape(1, C, 1, 1)
    beta = beta.reshape(1, C, 1, 1)

    mu = x.mean(axis=0)
    var = x.var(axis=0) + eps
    std = np.sqrt(var)
    z = (x - mu)/std
    z = z.T.reshape(N, C, H, W)
    out = gamma * z + beta

    cache={'std':std, 'gamma':gamma, 'z':z, 'size':size}

    pass

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return out, cache


def spatial_groupnorm_backward(dout, cache):
    """Computes the backward pass for spatial group normalization.

    Inputs:
    - dout: Upstream derivatives, of shape (N, C, H, W)
    - cache: Values from the forward pass

    Returns a tuple of:
    - dx: Gradient with respect to inputs, of shape (N, C, H, W)
    - dgamma: Gradient with respect to scale parameter, of shape (1, C, 1, 1)
    - dbeta: Gradient with respect to shift parameter, of shape (1, C, 1, 1)
    """
    dx, dgamma, dbeta = None, None, None

    ###########################################################################
    # TODO: Implement the backward pass for spatial group normalization.      #
    # This will be extremely similar to the layer norm implementation.        #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    N, C, H, W = dout.shape
    size = cache['size']
    dbeta = dout.sum(axis=(0,2,3), keepdims=True)
    dgamma = np.sum(dout * cache['z'], axis=(0,2,3), keepdims=True)


    z = cache['z'].reshape(size).T
    M = z.shape[0]
    dfdz = dout * cache['gamma']
    dfdz = dfdz.reshape(size).T

    dfdz_sum = np.sum(dfdz,axis=0)
    dx = dfdz - dfdz_sum/M - np.sum(dfdz * z,axis=0) * z/M
    dx /= cache['std']
    dx = dx.T.reshape(N, C, H, W)

    pass

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return dx, dgamma, dbeta
