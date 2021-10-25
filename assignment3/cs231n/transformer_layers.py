import torch
import torch.nn as nn
from torch.nn import functional as F
import math

"""
This file defines layer types that are commonly used for transformers.
"""

class PositionalEncoding(nn.Module):
    """
    Encodes information about the positions of the tokens in the sequence. In
    this case, the layer has no learnable parameters, since it is a simple
    function of sines and cosines.
    """
    def __init__(self, embed_dim, dropout=0.1, max_len=5000):
        """
        Construct the PositionalEncoding layer.

        Inputs:
         - embed_dim: the size of the embed dimension
         - dropout: the dropout value
         - max_len: the maximum possible length of the incoming sequence
        """
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        assert embed_dim % 2 == 0
        # Create an array with a "batch dimension" of 1 (which will broadcast
        # across all examples in the batch).
        pe = torch.zeros(1, max_len, embed_dim)
        ############################################################################
        # TODO: Construct the positional encoding array as described in            #
        # Transformer_Captioning.ipynb.  The goal is for each row to alternate     #
        # sine and cosine, and have exponents of 0, 0, 2, 2, 4, 4, etc. up to      #
        # embed_dim. Of course this exact specification is somewhat arbitrary, but #
        # this is what the autograder is expecting. For reference, our solution is #
        # less than 5 lines of code.                                               #
        ############################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

        pass

        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        # Make sure the positional encodings will be saved with the model
        # parameters (mostly for completeness).
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        Element-wise add positional embeddings to the input sequence.

        Inputs:
         - x: the sequence fed to the positional encoder model, of shape
              (N, S, D), where N is the batch size, S is the sequence length and
              D is embed dim
        Returns:
         - output: the input sequence + positional encodings, of shape (N, S, D)
        """
        N, S, D = x.shape
        # Create a placeholder, to be overwritten by your code below.
        output = torch.empty((N, S, D))
        ############################################################################
        # TODO: Index into your array of positional encodings, and add the         #
        # appropriate ones to the input sequence. Don't forget to apply dropout    #
        # afterward. This should only take a few lines of code.                    #
        ############################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        #print(self.pe.size())
        for i in range(S):
          for j in range(D):
            if j % 2 == 0 :
              self.pe[0, i, j] = math.sin(i * (10000 ** (-j/D)))
            else :
              self.pe[0, i, j] = math.cos(i * (10000 ** ((-j+1) / D)))
        #print(x.size())
        #print(self.pe.size())
        #print(x)
        output = x  + self.pe[:,0:S,:]
        output = self.dropout(output)
        #print(output)
        pass

        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################
        return output


class MultiHeadAttention(nn.Module):
    """
    A model layer which implements a simplified version of masked attention, as
    introduced by "Attention Is All You Need" (https://arxiv.org/abs/1706.03762).

    Usage:
      attn = MultiHeadAttention(embed_dim, num_heads=2)

      # self-attention
      data = torch.randn(batch_size, sequence_length, embed_dim)
      self_attn_output = attn(query=data, key=data, value=data)

      # attention using two inputs
      other_data = torch.randn(batch_size, sequence_length, embed_dim)
      attn_output = attn(query=data, key=other_data, value=other_data)
    """

    def __init__(self, embed_dim, num_heads, dropout=0.1):
        """
        Construct a new MultiHeadAttention layer.

        Inputs:
         - embed_dim: Dimension of the token embedding
         - num_heads: Number of attention heads
         - dropout: Dropout probability
        """
        super().__init__()
        assert embed_dim % num_heads == 0

        # We will initialize these layers for you, since swapping the ordering
        # would affect the random number generation (and therefore your exact
        # outputs relative to the autograder). Note that the layers use a bias
        # term, but this isn't strictly necessary (and varies by
        # implementation).
        ## 사용할 네트워크를 정의한다
        self.key = nn.Linear(embed_dim, embed_dim) ## nn.Linear에서 자동으로 weight이니셜라이즈 해주고 선형결합까지 해준다....
        self.query = nn.Linear(embed_dim, embed_dim)
        self.value = nn.Linear(embed_dim, embed_dim)
        self.proj = nn.Linear(embed_dim, embed_dim)

        
        ############################################################################
        # TODO: Initialize any remaining layers and parameters to perform the      #
        # attention operation as defined in Transformer_Captioning.ipynb. We will  #
        # also apply dropout just after the softmax step. For reference, our       #
        # solution is less than 5 lines.                                           #
        ############################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        self.scale_term = embed_dim // num_heads
        self.drop = nn.Dropout(p = dropout)
        self.num_heads = num_heads

        

        pass

        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

    def forward(self, query, key, value, attn_mask=None):
        """
        Calculate the masked attention output for the provided data, computing
        all attention heads in parallel.

        In the shape definitions below, N is the batch size, S is the source
        sequence length, T is the target sequence length, and E is the embedding
        dimension.

        Inputs:
        - query: Input data to be used as the query, of shape (N, S, E)
        - key: Input data to be used as the key, of shape (N, T, E)
        - value: Input data to be used as the value, of shape (N, T, E)
        - attn_mask: Array of shape (T, S) where mask[i,j] == 0 indicates token
          i in the target should not be influenced by token j in the source.

        Returns:
        - output: Tensor of shape (N, S, E) giving the weighted combination of
          data in value according to the attention weights calculated using key
          and query.
        """
        N, S, D = query.shape
        N, T, D = value.shape
        # Create a placeholder, to be overwritten by your code below.
        output = torch.empty((N, T, D))
        ############################################################################
        # TODO: Implement multiheaded attention using the equations given in       #
        # Transformer_Captioning.ipynb.                                            #
        # A few hints:                                                             #
        #  1) You'll want to split your shape from (N, T, E) into (N, T, H, E/H),  #
        #     where H is the number of heads.                                      #
        #  2) The function torch.matmul allows you to do a batched matrix multiply.#
        #     For example, you can do (N, H, T, E/H) by (N, H, E/H, T) to yield a  #
        #     shape (N, H, T, T). For more examples, see                           #
        #     https://pytorch.org/docs/stable/generated/torch.matmul.html          #
        #  3) For applying attn_mask, think how the scores should be modified to   #
        #     prevent a value from influencing output. Specifically, the PyTorch   #
        #     function masked_fill may come in handy.                              #
        ############################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        #print(query.size())
        #print(query)
        query = self.query(query) ## 이걸 해줘야 input query가 Wq와 선형결합한 아웃풋 값을 얻을수 있게된다
        query = query.reshape(N, S, self.num_heads, self.scale_term)
        key = self.key(key)
        key = key.reshape(N, T, self.num_heads, self.scale_term)
        value = self.value(value) ## 인풋으로 들어온 q, k, v 를 weight랑 선형결합 시켜준다
        value = value.reshape(N, T, self.num_heads, self.scale_term) ## head로 나눠서 쪼개준다.

        query = torch.transpose(query, 1, 2) ## seq to seq 계산을 위해 query의 head와 seq를 transpose해준다
        #print(query.size())
        key = torch.transpose(key, 1, 2)
        value = torch.transpose(value, 1, 2)

        query_key = torch.matmul(query, torch.transpose(key, -2,-1)) ## key를 transpose해줘서 seq to seq형태로 dot product해준다
        #print(query_key.size())
        soft = query_key / math.sqrt(self.scale_term) ## soft max 구분을 크게 하기위해 sqrt(scale_term)으로 나눠준다
        #print(soft.size())
        if attn_mask != None : ## mask 씌워줘야 할때 경우
          soft = soft.masked_fill(attn_mask == 0, -1e9) ## masked_fill을 이용해서 마스크가 mask가 0인 부분을 -1e9로 soft에 채워준다.
        score = F.softmax(soft, dim = -1) ## softmax 해준다.
        score = self.drop(score) ## drop out을 적용시켜준다.
        score_v = torch.matmul (score , value) ##softmax한 값을 value와 dotproduct해준다. 어텐션 energy를 value에 적용시켜주는 과정
        #print(N, T, D)
        #print(score_v.size())
        #concat = score_v.transpose(1,2).contiguous().view(N, T, D)
        concat = score_v.transpose(1,2) ##처음 모양대로 다시 트렌스 포즈해서 헤드와 시퀀스를 트랜스포즈해서 바꿔준다.
        #print(concat.size())
        concat = concat.reshape(N, S, D) ##이제 인풋과 같은 모양으로 reshape해준다.

        output = self.proj(concat)
        #print(output)




        pass

        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################
        return output


