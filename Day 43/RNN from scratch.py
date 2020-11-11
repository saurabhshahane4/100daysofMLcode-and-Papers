import d2l
import math
import torch
import torch.nn.functional as F
import torch.nn as nn
import time

corpus_indices, vocab = d2l.load_data_time_machine()

F.one_hot(torch.Tensor([0, 2]).long(), len(vocab))
def to_onehot(X,size):
    return F.one_hot(X.long().transpose(0,-1), size)

X = torch.arange(10).reshape((2, 5))
inputs = to_onehot(X, len(vocab))
len(inputs), inputs[0].shape

num_inputs, num_hiddens, num_outputs = len(vocab), 512, len(vocab)
ctx = d2l.try_gpu()
print('Using', ctx)

# Create the parameters of the model, initialize them and attach gradients
def get_params():
    def _one(shape):
        return torch.Tensor(size=shape, device=ctx).normal_(std=0.01)

    # Hidden layer parameters
    W_xh = _one((num_inputs, num_hiddens))
    W_hh = _one((num_hiddens, num_hiddens))
    b_h = torch.zeros(num_hiddens, device=ctx)
    # Output layer parameters
    W_hq = _one((num_hiddens, num_outputs))
    b_q = torch.zeros(num_outputs, device=ctx)
    # Attach a gradient
    params = [W_xh, W_hh, b_h, W_hq, b_q]
    for param in params:
        param.requires_grad_(True)
    return params

def init_rnn_state(batch_size, num_hiddens, ctx):
    return (torch.zeros(size=(batch_size, num_hiddens), device=ctx), )

def rnn(inputs, state, params):
    # Both inputs and outputs are composed of num_steps matrices of the shape
    # (batch_size, len(vocab))
    W_xh, W_hh, b_h, W_hq, b_q = params
    H, = state
    outputs = []
    for X in inputs:

        H = torch.tanh(torch.matmul(X.float(), W_xh) + torch.matmul(H.float(), W_hh) + b_h)
        Y = torch.matmul(H.float(), W_hq) + b_q
        outputs.append(Y)
    return outputs, (H,)

state = init_rnn_state(X.shape[0], num_hiddens, ctx)
inputs = to_onehot(X.to(ctx), len(vocab))
params = get_params()
outputs, state_new = rnn(inputs, state, params)
len(outputs), outputs[0].shape, state_new[0].shape

def predict_rnn(prefix, num_chars, rnn, params, init_rnn_state,
                num_hiddens, vocab, ctx):
    state = init_rnn_state(1, num_hiddens, ctx)
    output = [vocab[prefix[0]]]
    for t in range(num_chars + len(prefix) - 1):
        # The output of the previous time step is taken as the input of the
        # current time step.
        X = to_onehot(torch.Tensor([output[-1]],device=ctx), len(vocab))
        # Calculate the output and update the hidden state
        (Y, state) = rnn(X, state, params)
        # The input to the next time step is the character in the prefix or
        # the current best predicted character
        if t < len(prefix) - 1:
            # Read off from the given sequence of characters
            output.append(vocab[prefix[t + 1]])
        else:
            # This is maximum likelihood decoding. Modify this if you want
            # use sampling, beam search or beam sampling for better sequences.
            output.append(int(Y[0].argmax(dim=1).item()))
    return ''.join([vocab.idx_to_token[i] for i in output])

predict_rnn('traveller ', 10, rnn, params, init_rnn_state, num_hiddens,
            vocab, ctx)

def grad_clipping(params, theta, ctx):
    norm = torch.Tensor([0], device=ctx)
    for param in params:
        norm += (param.grad ** 2).sum()
    norm = norm.sqrt().item()
    if norm > theta:
        for param in params:
            param.grad.data.mul_(theta / norm)
def train_and_predict_rnn(rnn, get_params, init_rnn_state, num_hiddens,
                          corpus_indices, vocab, ctx, is_random_iter,
                          num_epochs, num_steps, lr, clipping_theta,
                          batch_size, prefixes):
    if is_random_iter:
        data_iter_fn = d2l.data_iter_random
    else:
        data_iter_fn = d2l.data_iter_consecutive
    params = get_params()
    loss =  nn.CrossEntropyLoss()
    start = time.time()
    for epoch in range(num_epochs):
        if not is_random_iter:
            # If adjacent sampling is used, the hidden state is initialized
            # at the beginning of the epoch
            state = init_rnn_state(batch_size, num_hiddens, ctx)
        l_sum, n = 0.0, 0
        data_iter = data_iter_fn(corpus_indices, batch_size, num_steps, ctx)
        for X, Y in data_iter:
            if is_random_iter:
                # If random sampling is used, the hidden state is initialized
                # before each mini-batch update
                state = init_rnn_state(batch_size, num_hiddens, ctx)
            else:
                # Otherwise, the detach function needs to be used to separate
                # the hidden state from the computational graph to avoid
                # backpropagation beyond the current sample
                for s in state:
                    s.detach_()
            inputs = to_onehot(X, len(vocab))
            # outputs is num_steps terms of shape (batch_size, len(vocab))
            (outputs, state) = rnn(inputs, state, params)
            # After stitching it is (num_steps * batch_size, len(vocab))
            outputs = torch.cat(outputs, dim=0)
            # The shape of Y is (batch_size, num_steps), and then becomes
            # a vector with a length of batch * num_steps after
            # transposition. This gives it a one-to-one correspondence
            # with output rows
            y = Y.t().reshape((-1,))
            # Average classification error via cross entropy loss
            l = loss(outputs, y.long()).mean()
            l.backward()
            with torch.no_grad():
                grad_clipping(params, clipping_theta, ctx)  # Clip the gradient
                d2l.sgd(params, lr, 1)
            # Since the error is the mean, no need to average gradients here
            l_sum += l.item() * y.numel()
            n += y.numel()
        if (epoch + 1) % 50 == 0:
            print('epoch %d, perplexity %f, time %.2f sec' % (
                epoch + 1, math.exp(l_sum / n), time.time() - start))
            start = time.time()
        if (epoch + 1) % 100 == 0:
            for prefix in prefixes:
                print(' -',  predict_rnn(prefix, 50, rnn, params,
                                         init_rnn_state, num_hiddens,
                                         vocab, ctx))

num_epochs, num_steps, batch_size, lr, clipping_theta = 500, 64, 32, 1, 1
prefixes = ['traveller', 'time traveller']     

train_and_predict_rnn(rnn, get_params, init_rnn_state, num_hiddens,
                      corpus_indices, vocab, ctx, True, num_epochs,
                      num_steps, lr, clipping_theta, batch_size, prefixes)