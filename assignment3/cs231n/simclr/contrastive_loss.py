import torch
import numpy as np


def sim(z_i, z_j):
    """Normalized dot product between two vectors.

    Inputs:
    - z_i: 1xD tensor.
    - z_j: 1xD tensor.
    
    Returns:
    - A scalar value that is the normalized dot product between z_i and z_j.
    """
    norm_dot_product = None
    ##############################################################################
    # TODO: Start of your code.                                                  #
    #                                                                            #
    # HINT: torch.linalg.norm might be helpful.                                  #
    ##############################################################################
    #z_j = torch.transpose(z_j, 0, 1)
    #print(z_i)
    dot = torch.dot(z_i, z_j)     # N * N dimension
    norm_z_i = torch.linalg.norm(z_i)   #scala
    norm_z_j = torch.linalg.norm(z_j)   #scala
    norm_dot_product = dot / (norm_z_i * norm_z_j)
    #print(norm_dot_product)
    
    ##############################################################################
    #                               END OF YOUR CODE                             #
    ##############################################################################
    
    return norm_dot_product


def simclr_loss_naive(out_left, out_right, tau):
    """Compute the contrastive loss L over a batch (naive loop version).
    
    Input:
    - out_left: NxD tensor; output of the projection head g(), left branch in SimCLR model.
    - out_right: NxD tensor; output of the projection head g(), right branch in SimCLR model.
    Each row is a z-vector for an augmented sample in the batch. The same row in out_left and out_right form a positive pair. 
    In other words, (out_left[k], out_right[k]) form a positive pair for all k=0...N-1.
    - tau: scalar value, temperature parameter that determines how fast the exponential increases.
    
    Returns:
    - A scalar value; the total loss across all positive pairs in the batch. See notebook for definition.
    """
    N = out_left.shape[0]  # total number of training examples
    
     # Concatenate out_left and out_right into a 2*N x D tensor.
    out = torch.cat([out_left, out_right], dim=0)  # [2*N, D]
    
    total_loss = 0
    for k in range(N):  # loop through each positive pair (k, k+N)
        z_k, z_k_N = out[k], out[k+N]
        
        ##############################################################################
        # TODO: Start of your code.                                                  #
        #                                                                            #
        # Hint: Compute l(k, k+N) and l(k+N, k).                                     #
        ##############################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        temp_sum_left = 0

        temp_left = (sim(out[k], out[k + N]) / tau).exp()

        for i in range(2 * N) :
          if i == k :
            continue
          
          temp_for_left = (sim(out[k], out[i]) / tau).exp()
          temp_sum_left += temp_for_left

        loss_left = -torch.log(temp_left / temp_sum_left)

        temp_sum_right = 0

        temp_right = (sim(out[k + N], out[k]) / tau).exp()

        for i in range(2 * N) :
          if i == k + N :
            continue
          
          temp_for_right = (sim(out[k + N], out[i]) / tau).exp()
          temp_sum_right += temp_for_right

        loss_right = -torch.log(temp_right / temp_sum_right)


        total_loss += (loss_left + loss_right)




        pass

        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
         ##############################################################################
        #                               END OF YOUR CODE                             #
        ##############################################################################
    
    # In the end, we need to divide the total loss by 2N, the number of samples in the batch.
    total_loss = total_loss / (2*N)
    return total_loss


def sim_positive_pairs(out_left, out_right):
    """Normalized dot product between positive pairs.

    Inputs:
    - out_left: NxD tensor; output of the projection head g(), left branch in SimCLR model.
    - out_right: NxD tensor; output of the projection head g(), right branch in SimCLR model.
    Each row is a z-vector for an augmented sample in the batch.
    The same row in out_left and out_right form a positive pair.
    
    Returns:
    - A Nx1 tensor; each row k is the normalized dot product between out_left[k] and out_right[k].
    """
    pos_pairs = None
    
    ##############################################################################
    # TODO: Start of your code.                                                  #
    #                                                                            #
    # HINT: torch.linalg.norm might be helpful.                                  #
    ##############################################################################
    
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    dot_temp = (out_left * out_right).sum(axis = 1, keepdim = True)
    #print(dot_temp)
    norm_left = torch.linalg.norm(out_left, dim = 1, keepdim = True)
    #print(norm_left.size())
    norm_right = torch.linalg.norm(out_right, dim = 1, keepdim = True)
    pos_pairs = dot_temp / (norm_left * norm_right)
    
    


    pass

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    
    ##############################################################################
    #                               END OF YOUR CODE                             #
    ##############################################################################
    return pos_pairs


def compute_sim_matrix(out):
    """Compute a 2N x 2N matrix of normalized dot products between all pairs of augmented examples in a batch.

    Inputs:
    - out: 2N x D tensor; each row is the z-vector (output of projection head) of a single augmented example.
    There are a total of 2N augmented examples in the batch.
    
    Returns:
    - sim_matrix: 2N x 2N tensor; each element i, j in the matrix is the normalized dot product between out[i] and out[j].
    """
    sim_matrix = None
    
    ##############################################################################
    # TODO: Start of your code.                                                  #
    ##############################################################################
    
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    dot_temp = torch.matmul(out, out.T)
    norm_temp = torch.linalg.norm(out, dim = 1, keepdim = True)
    norm_temp = torch.matmul(norm_temp, norm_temp.T)
    
    sim_matrix = dot_temp / norm_temp
    




    pass

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    
    ##############################################################################
    #                               END OF YOUR CODE                             #
    ##############################################################################
    return sim_matrix


def simclr_loss_vectorized(out_left, out_right, tau, device='cuda'):
    """Compute the contrastive loss L over a batch (vectorized version). No loops are allowed.
    
    Inputs and output are the same as in simclr_loss_naive.
    """
    N = out_left.shape[0]
    
    # Concatenate out_left and out_right into a 2*N x D tensor.
    out = torch.cat([out_left, out_right], dim=0)  # [2*N, D]
    
    # Compute similarity matrix between all pairs of augmented examples in the batch.
    sim_matrix = compute_sim_matrix(out)  # [2*N, 2*N]
    
    ##############################################################################
    # TODO: Start of your code. Follow the hints.                                #
    ##############################################################################
    
    # Step 1: Use sim_matrix to compute the denominator value for all augmented samples.
    # Hint: Compute e^{sim / tau} and store into exponential, which should have shape 2N x 2N.
    exponential = None
    exponential = torch.exp(sim_matrix / tau).to(device) ##오 쿠다로 안보내주면 오류나네
    #print(exponential)
    
    # This binary mask zeros out terms where k=i.
    mask = (torch.ones_like(exponential, device=device) - torch.eye(2 * N, device=device)).to(device).bool()
    #print(mask)
    
    # We apply the binary mask.
    exponential = exponential.masked_select(mask).view(2 * N, -1)  # [2*N, 2*N-1]
    #print(exponential)
    
    # Hint: Compute the denominator values for all augmented samples. This should be a 2N x 1 vector.
    denom = None


    # Step 2: Compute similarity between positive pairs.
    # You can do this in two ways: 
    # Option 1: Extract the corresponding indices from sim_matrix. 
    # Option 2: Use sim_positive_pairs().
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    denom = exponential.sum(dim = 1, keepdim = True).to(device) ##exponen이 중복되는 자기 자신을 뺀 행렬 2N x 2N-1 이니까 로우 기준으로 sum해주면 된다.
    #print(denom.size())

    #print(sum_denom)
    #print(positive_pair)


    pass

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    
    # Step 3: Compute the numerator value for all augmented samples.
    numerator = None
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    positive_pair_left = sim_positive_pairs(out_left, out_right).to(device) ## 분모에 해당하는 놈
    positive_pair_right = sim_positive_pairs(out_right, out_left).to(device) ## 좌우 바꿔서 한번씩 pair해줘서 sim을 해줘야 해서 왼쪽 오른쪽 한번씩
    numerator = torch.cat([positive_pair_left, positive_pair_right], dim = 0) ## 왼쪽 오른쪽 해준 녀석을 concat해준다 그리하여 2N x 1로

    numerator = (numerator / tau).exp() ## 분모에 tau로 나눠주고 exp적용
    #print(numerator.size())

    pass

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    
    # Step 4: Now that you have the numerator and denominator for all augmented samples, compute the total loss.
    loss = None

    loss =  -torch.log(numerator / denom)

    loss = loss.sum()

    loss = loss / (2*N)
    
    ##############################################################################
    #                               END OF YOUR CODE                             #
    ##############################################################################
    
    return loss

# *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

pass

# *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

def rel_error(x,y):
    return np.max(np.abs(x - y) / (np.maximum(1e-8, np.abs(x) + np.abs(y))))