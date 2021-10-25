from builtins import range
import numpy as np
from random import shuffle
from past.builtins import xrange


def svm_loss_naive(W, X, y, reg):
    """
    Structured SVM loss function, naive implementation (with loops).

    Inputs have dimension D, there are C classes, and we operate on minibatches
    of N examples.

    Inputs:
    - W: A numpy array of shape (D, C) containing weights.
    - X: A numpy array of shape (N, D) containing a minibatch of data.
    - y: A numpy array of shape (N,) containing training labels; y[i] = c means
      that X[i] has label c, where 0 <= c < C.
    - reg: (float) regularization strength

    Returns a tuple of:
    - loss as single float
    - gradient with respect to weights W; an array of same shape as W
    """
    dW = np.zeros(W.shape)  # initialize the gradient as zero

    # compute the loss and the gradient
    num_classes = W.shape[1] # 클래스 수는 10개
    num_train = X.shape[0] # 트레인 셋 수는 dev X니까 500개
    loss = 0.0
    for i in range(num_train):
        scores = X[i].dot(W)
        correct_class_score = scores[y[i]] # 위에서 구한 스코어 리스트의 정답에 해당하는 스코어를 콜랙트클래스스코어 변수에 저장해준다
        for j in range(num_classes):
            if j == y[i]: #우리가 비교하려는 데이터와 같은 클래스에 대해서는 그냥 건너뛰어라 마진을 더하지 않고
                continue
            margin = scores[j] - correct_class_score + 1  # note delta = 1
            if margin > 0:
                loss += margin
                

    # Right now the loss is a sum over all training examples, but we want it
    # to be an average instead so we divide by num_train.
    loss /= num_train # 모든 트레이닝 예제에 대해 더해진 로스를 트레인셋 수만큼 나눠 평균을 구한다
    # Add regularization to the loss.
    #loss += reg * np.sum(W * W) #정규화항 추가 reg는 정규화항 하이퍼파라미터

    #############################################################################
    # TODO:                                                                     #
    # Compute the gradient of the loss function and store it dW.                #
    # Rather that first computing the loss and then computing the derivative,   #
    # it may be simpler to compute the derivative at the same time that the     #
    # loss is being computed. As a result you may need to modify some of the    #
    # code above to compute the gradient.                                       #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    loss = loss + ((reg * 0.5) * (np.sum(W * W)))
    for i in range(num_train):
        scores = X[i].dot(W)
        correct_class_score = scores[y[i]] # 위에서 구한 스코어 리스트의 정답에 해당하는 스코어를 콜랙트클래스스코어 변수에 저장해준다
        for j in range(num_classes):
            if j == y[i]: #우리가 비교하려는 데이터와 같은 클래스에 대해서는 그냥 건너뛰어라 마진을 더하지 않고
                continue
            margin = scores[j] - correct_class_score + 1  # note delta = 1
            if margin > 0:
                dW[:,j] = dW[:,j] + X[i] # dW의 다른레이블에 대한 도함수는 x니까 더해준다
                dW[:,y[i]] = dW[:,y[i]] - X[i]
    dW = dW / num_train #트레이닝셋수만큼 더하였으니 트레이닝셋 수만큼 나눠 평균을 낸다
    dW = dW + (reg * W) #L2 reguralize 의 도함수를 더해준다


    pass

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW


def svm_loss_vectorized(W, X, y, reg):
    """
    Structured SVM loss function, vectorized implementation.

    Inputs and outputs are the same as svm_loss_naive.
    """
    loss = 0.0
    dW = np.zeros(W.shape)  # initialize the gradient as zero

    #############################################################################
    # TODO:                                                                     #
    # Implement a vectorized version of the structured SVM loss, storing the    #
    # result in loss.                                                           #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    # score구한다
    scores = np.dot(X,W)
    # 정답레이블에 해당되는 score 만 따로 뽑아낸다
    yi_scores = scores[np.arange(scores.shape[0]), y]
    # 음수는 maximum으로 버려버리고 score와 정답레이블 score를 broadcast로 빼준고 safty margin을 더해준다
    margins = np.maximum(0, scores - np.matrix(yi_scores).T + 1)
    # 정답 레이블에 해당되는 놈들은 0으로 처리해버린다.
    margins[np.arange(X.shape[0]), y] = 0
    # 로스구하고
    loss = np.mean(np.sum(margins, axis = 1))
    # regularization 항 추가해준다.
    loss += (0.5 * reg) + np.sum(W * W)

    pass

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    #############################################################################
    # TODO:                                                                     #
    # Implement a vectorized version of the gradient for the structured SVM     #
    # loss, storing the result in dW.                                           #
    #                                                                           #
    # Hint: Instead of computing the gradient from scratch, it may be easier    #
    # to reuse some of the intermediate values that you used to compute the     #
    # loss.                                                                     #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    #위에 우리가 구한 마진을 템프에다 임시저장
    temp = margins
    #마진이 0보다 큰 놈들은 미분하면 전부 1이니까 1로 변경
    temp[margins > 0] = 1
    row_sum = np.sum(temp, axis = 1)
    #정답 레이블에 해당되는 놈들은 음수값으로 할당
    temp[np.arange(X.shape[0]), y] = -row_sum.T
    dW = np.dot(X.T,temp)

    dW /= X.shape[0]
    dW += reg * W

    pass

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW
