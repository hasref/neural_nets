from __future__ import print_function, division
from builtins import range
import numpy as np


"""
This file defines layer types that are commonly used for recurrent neural
networks.
"""


def rnn_step_forward(x, prev_h, Wx, Wh, b):
    """
    Run the forward pass for a single timestep of a vanilla RNN that uses a tanh
    activation function.

    The input data has dimension D, the hidden state has dimension H, and we use
    a minibatch size of N.

    Inputs:
    - x: Input data for this timestep, of shape (N, D).
    - prev_h: Hidden state from previous timestep, of shape (N, H)
    - Wx: Weight matrix for input-to-hidden connections, of shape (D, H)
    - Wh: Weight matrix for hidden-to-hidden connections, of shape (H, H)
    - b: Biases of shape (H,)

    Returns a tuple of:
    - next_h: Next hidden state, of shape (N, H)
    - cache: Tuple of values needed for the backward pass.
    """
    next_h, cache = None, None
    ##############################################################################
    # TODO: Implement a single forward step for the vanilla RNN. Store the next  #
    # hidden state and any values you need for the backward pass in the next_h   #
    # and cache variables respectively.                                          #
    ##############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    next_h = np.tanh( x.dot(Wx) + prev_h.dot(Wh) + b )
    cache = (x, prev_h, Wx, Wh, b, next_h )
    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ##############################################################################
    #                               END OF YOUR CODE                             #
    ##############################################################################
    return next_h, cache


def rnn_step_backward(dnext_h, cache):
    """
    Backward pass for a single timestep of a vanilla RNN.

    Inputs:
    - dnext_h: Gradient of loss with respect to next hidden state, of shape (N, H)
    - cache: Cache object from the forward pass

    Returns a tuple of:
    - dx: Gradients of input data, of shape (N, D)
    - dprev_h: Gradients of previous hidden state, of shape (N, H)
    - dWx: Gradients of input-to-hidden weights, of shape (D, H)
    - dWh: Gradients of hidden-to-hidden weights, of shape (H, H)
    - db: Gradients of bias vector, of shape (H,)
    """
    dx, dprev_h, dWx, dWh, db = None, None, None, None, None
    ##############################################################################
    # TODO: Implement the backward pass for a single step of a vanilla RNN.      #
    #                                                                            #
    # HINT: For the tanh function, you can compute the local derivative in terms #
    # of the output value from tanh.                                             #
    ##############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    x, prev_h, Wx, Wh, b, next_h = cache
    
    #backpropr throuogh tanh dtanh = 1 - tanh**2
    dtanh = 1 - next_h * next_h
    dtanh = dnext_h * dtanh
    
    db = np.sum(dtanh, axis=0)
    dWx = np.dot( x.T, dtanh )
    dWh = np.dot ( prev_h.T, dtanh )
    
    dx = dtanh.dot( Wx.T )
    dprev_h = dtanh.dot ( Wh.T )

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ##############################################################################
    #                               END OF YOUR CODE                             #
    ##############################################################################
    return dx, dprev_h, dWx, dWh, db


def rnn_forward(x, h0, Wx, Wh, b):
    """
    Run a vanilla RNN forward on an entire sequence of data. We assume an input
    sequence composed of T vectors, each of dimension D. The RNN uses a hidden
    size of H, and we work over a minibatch containing N sequences. After running
    the RNN forward, we return the hidden states for all timesteps.

    Inputs:
    - x: Input data for the entire timeseries, of shape (N, T, D).
    - h0: Initial hidden state, of shape (N, H)
    - Wx: Weight matrix for input-to-hidden connections, of shape (D, H)
    - Wh: Weight matrix for hidden-to-hidden connections, of shape (H, H)
    - b: Biases of shape (H,)

    Returns a tuple of:
    - h: Hidden states for the entire timeseries, of shape (N, T, H).
    - cache: Values needed in the backward pass
    """
    h, cache = None, None
    ##############################################################################
    # TODO: Implement forward pass for a vanilla RNN running on a sequence of    #
    # input data. You should use the rnn_step_forward function that you defined  #
    # above. You can use a for loop to help compute the forward pass.            #
    ##############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    N,T,D = x.shape
    _, H = Wh.shape
    
    h = np.zeros( (N, T, H ) )
    prev_h = h0  
    for t in range(T):
        h[:, t, :], _ = rnn_step_forward( x[:, t, :], prev_h, Wx, Wh, b)
        prev_h = h[:, t, :]
        
    cache = x, h0, h, Wx, Wh, b
    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ##############################################################################
    #                               END OF YOUR CODE                             #
    ##############################################################################
    return h, cache


def rnn_backward(dh, cache):
    """
    Compute the backward pass for a vanilla RNN over an entire sequence of data.

    Inputs:
    - dh: Upstream gradients of all hidden states, of shape (N, T, H). 
    
    NOTE: 'dh' contains the upstream gradients produced by the 
    individual loss functions at each timestep, *not* the gradients
    being passed between timesteps (which you'll have to compute yourself
    by calling rnn_step_backward in a loop).

    Returns a tuple of:
    - dx: Gradient of inputs, of shape (N, T, D)
    - dh0: Gradient of initial hidden state, of shape (N, H)
    - dWx: Gradient of input-to-hidden weights, of shape (D, H)
    - dWh: Gradient of hidden-to-hidden weights, of shape (H, H)
    - db: Gradient of biases, of shape (H,)
    """
    dx, dh0, dWx, dWh, db = None, None, None, None, None
    ##############################################################################
    # TODO: Implement the backward pass for a vanilla RNN running an entire      #
    # sequence of data. You should use the rnn_step_backward function that you   #
    # defined above. You can use a for loop to help compute the backward pass.   #
    ##############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    x, h0, h, Wx, Wh, b = cache
    
    N,T,H = x.shape
    D,_ = Wx.shape
    
    dx = np.zeros(np.shape(x) )
    dh0 = np.zeros(np.shape(h0) )
    dWx = np.zeros(np.shape(Wx) )
    dWh = np.zeros(np.shape(Wh) )
    db = np.zeros(np.shape(b) )
    dprev_h = np.zeros( np.shape(h0) )
    
    # do last step outside
    for t in range(T-1, 0 ,-1):
        # dh only contains gradient of h with respect to local loss
        # add value of gradient flowing from upstream --> how does the 
        # the hidden state output at this step, affect the loss at future time 
        # steps
        dnext_h = dh[:, t, :] + dprev_h
        
        dback = rnn_step_backward( dnext_h, ( x[:, t, :], h[:, t-1, :], Wx, Wh, 
                                     b, h[:,t,:] ) )
        
        dx_t, dprev_h, dWx_t, dWh_t, db_t = dback
        
        dx[:, t, :] = dx_t
        
        # parameter sharing --> gradients add
        dWx += dWx_t
        dWh += dWh_t
        db += db_t
    
    
    # last iteration
    dnext_h = dh[:, 0 , :] + dprev_h
    dback = rnn_step_backward( dnext_h, ( x[:, 0, :], h0 , Wx, Wh, 
                                 b, h[:, 0 ,:] ) )
    
    dx_t, dprev_h, dWx_t, dWh_t, db_t = dback
    
    dx[:, 0, :] = dx_t
    dWx += dWx_t
    dWh += dWh_t
    db += db_t
    
    # dh0 is the gradient of the total loss with respect to the initial hidden
    # state
    dh0 = dprev_h
    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ##############################################################################
    #                               END OF YOUR CODE                             #
    ##############################################################################
    return dx, dh0, dWx, dWh, db


def word_embedding_forward(x, W):
    """
    Forward pass for word embeddings. We operate on minibatches of size N where
    each sequence has length T. We assume a vocabulary of V words, assigning each
    word to a vector of dimension D.

    Inputs:
    - x: Integer array of shape (N, T) giving indices of words. Each element idx
      of x muxt be in the range 0 <= idx < V.
    - W: Weight matrix of shape (V, D) giving word vectors for all words.

    Returns a tuple of:
    - out: Array of shape (N, T, D) giving word vectors for all input words.
    - cache: Values needed for the backward pass
    """
    out, cache = None, None
    ##############################################################################
    # TODO: Implement the forward pass for word embeddings.                      #
    #                                                                            #
    # HINT: This can be done in one line using NumPy's array indexing.           #
    ##############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    out = W[x,:]
    cache = (x,W)

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ##############################################################################
    #                               END OF YOUR CODE                             #
    ##############################################################################
    return out, cache


def word_embedding_backward(dout, cache):
    """
    Backward pass for word embeddings. We cannot back-propagate into the words
    since they are integers, so we only return gradient for the word embedding
    matrix.

    HINT: Look up the function np.add.at

    Inputs:
    - dout: Upstream gradients of shape (N, T, D)
    - cache: Values from the forward pass

    Returns:
    - dW: Gradient of word embedding matrix, of shape (V, D).
    """
    dW = None
    ##############################################################################
    # TODO: Implement the backward pass for word embeddings.                     #
    #                                                                            #
    # Note that words can appear more than once in a sequence.                   #
    # HINT: Look up the function np.add.at                                       #
    ##############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    x,W = cache
    
    V, _ = W.shape
    N,T,D = dout.shape
    
    dW = np.zeros( (V, D) )
    
    # for each "idx" in x, we want to accumulate the derivatives in dout at 
    # dW[ idx, : ] since this would then give us the gradient for each word 
    # embedding vector
    
    for row in range(N):
        for column in range(T):
            dW[ x[row, column], : ] += dout[ row, column, : ]

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ##############################################################################
    #                               END OF YOUR CODE                             #
    ##############################################################################
    return dW


def sigmoid(x):
    """
    A numerically stable version of the logistic sigmoid function.
    """
    pos_mask = (x >= 0)
    neg_mask = (x < 0)
    z = np.zeros_like(x)
    z[pos_mask] = np.exp(-x[pos_mask])
    z[neg_mask] = np.exp(x[neg_mask])
    top = np.ones_like(x)
    top[neg_mask] = z[neg_mask]
    return top / (1 + z)


def lstm_step_forward(x, prev_h, prev_c, Wx, Wh, b):
    """
    Forward pass for a single timestep of an LSTM.

    The input data has dimension D, the hidden state has dimension H, and we use
    a minibatch size of N.

    Note that a sigmoid() function has already been provided for you in this file.

    Inputs:
    - x: Input data, of shape (N, D)
    - prev_h: Previous hidden state, of shape (N, H)
    - prev_c: previous cell state, of shape (N, H)
    - Wx: Input-to-hidden weights, of shape (D, 4H)
    - Wh: Hidden-to-hidden weights, of shape (H, 4H)
    - b: Biases, of shape (4H,)

    Returns a tuple of:
    - next_h: Next hidden state, of shape (N, H)
    - next_c: Next cell state, of shape (N, H)
    - cache: Tuple of values needed for backward pass.
    """
    next_h, next_c, cache = None, None, None
    #############################################################################
    # TODO: Implement the forward pass for a single timestep of an LSTM.        #
    # You may want to use the numerically stable sigmoid implementation above.  #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    N, H = prev_h.shape
    
    activations = x.dot(Wx) + prev_h.dot(Wh) + b
    
    sigmoid_activations = sigmoid( activations[:, :3*H] )
    
    input_gate =  sigmoid_activations [:, :H ] 
    forget_gate = sigmoid_activations [:, H:2*H]     
    output_gate = sigmoid_activations [:, 2*H : 3*H]
    
    # Justin calls this the "gate gate"
    gate_gate = np.tanh ( activations[:, 3*H: ] )
    
    next_c = forget_gate * prev_c + input_gate * gate_gate
    next_h = output_gate * np.tanh( next_c )
    
    cache = (x, prev_h, prev_c, next_c, next_h, input_gate, forget_gate,
             output_gate, gate_gate, Wx, Wh, b)

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ##############################################################################
    #                               END OF YOUR CODE                             #
    ##############################################################################

    return next_h, next_c, cache


def lstm_step_backward(dnext_h, dnext_c, cache, **kwargs):
    """
    Backward pass for a single timestep of an LSTM.

    Inputs:
    - dnext_h: Gradients of next hidden state, of shape (N, H)
    - dnext_c: Gradients of next cell state, of shape (N, H)
    - cache: Values from the forward pass

    Returns a tuple of:
    - dx: Gradient of input data, of shape (N, D)
    - dprev_h: Gradient of previous hidden state, of shape (N, H)
    - dprev_c: Gradient of previous cell state, of shape (N, H)
    - dWx: Gradient of input-to-hidden weights, of shape (D, 4H)
    - dWh: Gradient of hidden-to-hidden weights, of shape (H, 4H)
    - db: Gradient of biases, of shape (4H,)
    """
    dx, dprev_h, dprev_c, dWx, dWh, db = None, None, None, None, None, None
    #############################################################################
    # TODO: Implement the backward pass for a single timestep of an LSTM.       #
    #                                                                           #
    # HINT: For sigmoid and tanh you can compute local derivatives in terms of  #
    # the output value from the nonlinearity.                                   #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    x, prev_h, prev_c, next_c, next_h, input_gate, forget_gate, \
             output_gate, gate_gate, Wx, Wh, b = cache

    # backprop through hidden state equation
    doutput_gate = np.tanh(next_c) * dnext_h
    
    # add kwargs to allow lstm_backward function to rely on our step function
    kwargs.setdefault('time_step', 'regular')
    
    if ( kwargs['time_step'] == 'regular' ):
        dnext_c_through_h = output_gate *  ( 1 - np.tanh(next_c) * np.tanh(next_c) )\
        * dnext_h
    else:
        dnext_c_through_h = np.zeros( np.shape(dnext_c) )
    
    dnext_c += dnext_c_through_h             
    
    # back prop through cell state equation
    dforget_gate = prev_c * dnext_c
    dinput_gate = gate_gate * dnext_c
    dgate_gate = input_gate * dnext_c
    dprev_c = forget_gate * dnext_c
    
    # backprop through input, forget, output, and "gate" gates
    # first three through sigmoid gate
    dactiv_input = input_gate * (1 - input_gate) * dinput_gate
    dactiv_forget = forget_gate * (1 - forget_gate) * dforget_gate
    dactiv_output = output_gate * (1 - output_gate) * doutput_gate
    
    # through tanh gate
    dactiv_gate = dgate_gate * ( 1 - gate_gate ** 2 )
    
    #print( dactiv_input.shape, dactiv_forget.shape, dactiv_output.shape, dactiv_gate.shape)
    
    dactiv = np.hstack ( (dactiv_input, dactiv_forget, dactiv_output, 
                        dactiv_gate ) ) # N, 4H
    
    # backprop through activation affine layer
    db = np.sum(dactiv, axis=0)
    
    dx = dactiv.dot( Wx.T )
    dprev_h = dactiv.dot( Wh.T )
    
    dWx = (x.T).dot(dactiv)
    dWh = (prev_h.T).dot(dactiv)
    
    

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ##############################################################################
    #                               END OF YOUR CODE                             #
    ##############################################################################

    return dx, dprev_h, dprev_c, dWx, dWh, db


def lstm_forward(x, h0, Wx, Wh, b):
    """
    Forward pass for an LSTM over an entire sequence of data. We assume an input
    sequence composed of T vectors, each of dimension D. The LSTM uses a hidden
    size of H, and we work over a minibatch containing N sequences. After running
    the LSTM forward, we return the hidden states for all timesteps.

    Note that the initial cell state is passed as input, but the initial cell
    state is set to zero. Also note that the cell state is not returned; it is
    an internal variable to the LSTM and is not accessed from outside.

    Inputs:
    - x: Input data of shape (N, T, D)
    - h0: Initial hidden state of shape (N, H)
    - Wx: Weights for input-to-hidden connections, of shape (D, 4H)
    - Wh: Weights for hidden-to-hidden connections, of shape (H, 4H)
    - b: Biases of shape (4H,)

    Returns a tuple of:
    - h: Hidden states for all timesteps of all sequences, of shape (N, T, H)
    - cache: Values needed for the backward pass.
    """
    h, cache = None, None
    #############################################################################
    # TODO: Implement the forward pass for an LSTM over an entire timeseries.   #
    # You should use the lstm_step_forward function that you just defined.      #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    
    _, T, D = x.shape
    N,H = h0.shape

    # long term storage for backprop
    hidden_states = np.zeros( (N, T+1, H) )
    cell_states = np.zeros( (N, T+1, H) )
    
    input_gates = np.zeros( ( N, T, H ) )
    output_gates = np.zeros( ( N, T, H ) )
    forget_gates = np.zeros ( ( N, T, H ) )
    gate_gates = np.zeros ( ( N, T, H ) )


    hidden_states[:, 0, :] = h0
    cell_states [:, 0, :] = np.zeros( ( N, H ) )
        
    for t in range (T): 
        h , c , c_t = lstm_step_forward( x[:, t, :], 
                        hidden_states[:, t, :], cell_states[:, t, :],
                        Wx, Wh, b )
        
        _, _, _, _ , _ , i_gate, f_gate, o_gate, g_gate, _, _, _ = c_t

        # storage is offset by one since the first (t = 0) entry is h0 and c0
        hidden_states[:, t+1, :] = h
        cell_states[:, t+1, :] = c
        
        input_gates[:, t, :] = i_gate
        output_gates [:, t, : ] = o_gate
        forget_gates [:, t, :] = f_gate
        gate_gates [:, t, :] = g_gate
        
        
    cache = (x, hidden_states, cell_states, input_gates, output_gates, 
             forget_gates, gate_gates, Wx, Wh, b)
    
    h = hidden_states[:, 1:, :]
    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ##############################################################################
    #                               END OF YOUR CODE                             #
    ##############################################################################

    return h, cache


def lstm_backward(dh, cache):
    """
    Backward pass for an LSTM over an entire sequence of data.]

    Inputs:
    - dh: Upstream gradients of hidden states, of shape (N, T, H)
    - cache: Values from the forward pass

    Returns a tuple of:
    - dx: Gradient of input data of shape (N, T, D)
    - dh0: Gradient of initial hidden state of shape (N, H)
    - dWx: Gradient of input-to-hidden weight matrix of shape (D, 4H)
    - dWh: Gradient of hidden-to-hidden weight matrix of shape (H, 4H)
    - db: Gradient of biases, of shape (4H,)
    """
    dx, dh0, dWx, dWh, db = None, None, None, None, None
    #############################################################################
    # TODO: Implement the backward pass for an LSTM over an entire timeseries.  #
    # You should use the lstm_step_backward function that you just defined.     #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    x, hidden_states, cell_states, input_gates, output_gates, \
         forget_gates, gate_gates, Wx, Wh, b = cache
         
    N, T, H = dh.shape
    _ , _, D = x.shape
    
    dx = np.zeros( x.shape )
    db = np.zeros( b.shape )
    dWh = np.zeros( Wh.shape )
    dWx = np.zeros ( Wx.shape )
    
    # initial cell derivative comes from dh
    final_cell_state = cell_states[:, T, :]
    c = np.tanh(final_cell_state)
    
    dc_t = output_gates[:, T-1, :] *  ( 1 - c * c) * dh[ :, T-1 , : ]
    dh_t = np.zeros( ( N,H ) )
    
    time = 'final'
    for t in range(T-1, -1, -1):
        # construct the cache for the time step t
        cache_t = (x[:, t, :], hidden_states[:, t, :], cell_states[:, t, :], 
               cell_states[:, t+1, :], hidden_states[:, t+1, :], 
               input_gates[:, t,:], forget_gates[:,t,:], output_gates[:, t, :],
               gate_gates[:,t,:], Wx, Wh, b)
        
        # dh only contains derivatives with respect to local loss function. 
        # the upstream gradient for the hidden state must therefore be added 
        # to it ---> dh[:,t,:] + dh_t
        
        dx_t, dh_t, dc_t, dWx_t, dWh_t, db_t =  lstm_step_backward( dh[:,t,:] + dh_t, 
                                                                   dc_t, cache_t,
                                                                   time_step=time)
        
        # accumulate gradients
        dWx += dWx_t
        dWh += dWh_t
        db  += db_t
        dx[:,t,:] = dx_t
        # set time_step to regular for all other time steps
        time = 'regular'
    
    # last dh_t is dh0
    dh0 = dh_t

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ##############################################################################
    #                               END OF YOUR CODE                             #
    ##############################################################################

    return dx, dh0, dWx, dWh, db


def temporal_affine_forward(x, w, b):
    """
    Forward pass for a temporal affine layer. The input is a set of D-dimensional
    vectors arranged into a minibatch of N timeseries, each of length T. We use
    an affine function to transform each of those vectors into a new vector of
    dimension M.

    Inputs:
    - x: Input data of shape (N, T, D)
    - w: Weights of shape (D, M)
    - b: Biases of shape (M,)

    Returns a tuple of:
    - out: Output data of shape (N, T, M)
    - cache: Values needed for the backward pass
    """
    N, T, D = x.shape
    M = b.shape[0]
    out = x.reshape(N * T, D).dot(w).reshape(N, T, M) + b
    cache = x, w, b, out
    return out, cache


def temporal_affine_backward(dout, cache):
    """
    Backward pass for temporal affine layer.

    Input:
    - dout: Upstream gradients of shape (N, T, M)
    - cache: Values from forward pass

    Returns a tuple of:
    - dx: Gradient of input, of shape (N, T, D)
    - dw: Gradient of weights, of shape (D, M)
    - db: Gradient of biases, of shape (M,)
    """
    x, w, b, out = cache
    N, T, D = x.shape
    M = b.shape[0]

    dx = dout.reshape(N * T, M).dot(w.T).reshape(N, T, D)
    dw = dout.reshape(N * T, M).T.dot(x.reshape(N * T, D)).T
    db = dout.sum(axis=(0, 1))

    return dx, dw, db


def temporal_softmax_loss(x, y, mask, verbose=False):
    """
    A temporal version of softmax loss for use in RNNs. We assume that we are
    making predictions over a vocabulary of size V for each timestep of a
    timeseries of length T, over a minibatch of size N. The input x gives scores
    for all vocabulary elements at all timesteps, and y gives the indices of the
    ground-truth element at each timestep. We use a cross-entropy loss at each
    timestep, summing the loss over all timesteps and averaging across the
    minibatch.

    As an additional complication, we may want to ignore the model output at some
    timesteps, since sequences of different length may have been combined into a
    minibatch and padded with NULL tokens. The optional mask argument tells us
    which elements should contribute to the loss.

    Inputs:
    - x: Input scores, of shape (N, T, V)
    - y: Ground-truth indices, of shape (N, T) where each element is in the range
         0 <= y[i, t] < V
    - mask: Boolean array of shape (N, T) where mask[i, t] tells whether or not
      the scores at x[i, t] should contribute to the loss.

    Returns a tuple of:
    - loss: Scalar giving loss
    - dx: Gradient of loss with respect to scores x.
    """

    N, T, V = x.shape

    x_flat = x.reshape(N * T, V)
    y_flat = y.reshape(N * T)
    mask_flat = mask.reshape(N * T)

    probs = np.exp(x_flat - np.max(x_flat, axis=1, keepdims=True))
    probs /= np.sum(probs, axis=1, keepdims=True)
    loss = -np.sum(mask_flat * np.log(probs[np.arange(N * T), y_flat])) / N
    dx_flat = probs.copy()
    dx_flat[np.arange(N * T), y_flat] -= 1
    dx_flat /= N
    dx_flat *= mask_flat[:, None]

    if verbose: print('dx_flat: ', dx_flat.shape)

    dx = dx_flat.reshape(N, T, V)

    return loss, dx
