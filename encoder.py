import time
import numpy as np
import tensorflow as tf

from utils import HParams, preprocess, iter_data

global nloaded
nloaded = 0


def load_params(shape, dtype, *args, **kwargs):
    global nloaded
    nloaded += 1
    return params[nloaded - 1]


def embd(X, ndim, scope='embedding'):
    with tf.variable_scope(scope):
        embd = tf.get_variable(
            "w", [hps.nvocab, ndim], initializer=load_params)
        h = tf.nn.embedding_lookup(embd, X)
        return h

def mlstm(inputs, c, h, M, ndim, scope='lstm', wn=False):
    nin = inputs[0].get_shape()[1].value
    with tf.variable_scope(scope):
        wx = tf.get_variable("wx", [nin, ndim * 4], initializer=load_params)
        wh = tf.get_variable("wh", [ndim, ndim * 4], initializer=load_params)
        wmx = tf.get_variable("wmx", [nin, ndim], initializer=load_params)
        wmh = tf.get_variable("wmh", [ndim, ndim], initializer=load_params)
        b = tf.get_variable("b", [ndim * 4], initializer=load_params)
        if wn:
            gx = tf.get_variable("gx", [ndim * 4], initializer=load_params)
            gh = tf.get_variable("gh", [ndim * 4], initializer=load_params)
            gmx = tf.get_variable("gmx", [ndim], initializer=load_params)
            gmh = tf.get_variable("gmh", [ndim], initializer=load_params)

    if wn:
        wx = tf.nn.l2_normalize(wx, dim=0) * gx
        wh = tf.nn.l2_normalize(wh, dim=0) * gh
        wmx = tf.nn.l2_normalize(wmx, dim=0) * gmx
        wmh = tf.nn.l2_normalize(wmh, dim=0) * gmh

    for idx, x in enumerate(inputs):
        m = tf.matmul(x, wmx)*tf.matmul(h, wmh)
        z = tf.matmul(x, wx) + tf.matmul(m, wh) + b
        i, f, o, u = tf.split(z, num_or_size_splits=4, axis=1)
        i = tf.nn.sigmoid(i)
        f = tf.nn.sigmoid(f)
        o = tf.nn.sigmoid(o)
        u = tf.tanh(u)
        # if M is None we have pure LSTM
        if M is not None:
            ct = f*c + i*u
            ht = o*tf.tanh(ct)
            m = M[:, idx, :]
            c = ct*m + c*(1-m)
            h = ht*m + h*(1-m)
        else:
            c = f*c + i*u
            h = o*tf.tanh(c)
        inputs[idx] = h
    return c, h


def model(X, S, M=None, reuse=False):
    """
    Unstack internal and hidden states, run mLSTM and stack back

    :param X: input sentences represented as sequences of ASCII values
    :param S: outputs?
    :param M: binary encoding of sentences
    :param reuse:
    :return states: updated S
    """
    nsteps = X.get_shape()[1].value
    cstart, hstart = tf.unstack(S, num=hps.nstates) # c - internal state, h - hidden state, M - intermediate state
    with tf.variable_scope('model', reuse=reuse):
        words = embd(X, hps.nembd)
        inputs = [tf.squeeze(v, [1]) for v in tf.split(words, num_or_size_splits=nsteps, axis=1)]
        cfinal, hfinal = mlstm(inputs, cstart, hstart, M, hps.nhidden, scope='rnn', wn=hps.rnn_wn)
    states = tf.stack([cfinal, hfinal], 0)
    return states


def ceil_round_step(n, step):
    return int(np.ceil(n/step)*step)


def batch_pad(xs, nbatch, nsteps):
    """
    Input to model.transform(["text"]) is output of this function.
    :param xs: list of sentences, each encoded as a sequence of ASCII values
    :param nbatch: number of input sentences?
    :param nsteps: 64, what is this?
    :return xmb, mmb: sentences as sequences of ASCII values, binary indicators of presence of char
    """
    xmb = np.zeros((nbatch, nsteps), dtype=np.int32)
    mmb = np.ones((nbatch, nsteps, 1), dtype=np.float32)
    for i, x in enumerate(xs):
        l = len(x)
        npad = nsteps-l
        xmb[i, -l:] = list(x) # ASCII values of each character, filled starting from the end
        mmb[i, :npad] = 0 # 1 - if there is a symbol in xmb, 0 if no
    return xmb, mmb


class Model(object):

    def __init__(self, nbatch=128, nsteps=64):
        global hps
        hps = HParams(
            load_path='model_params/params.jl',
            nhidden=4096,
            nembd=64,
            nsteps=nsteps,
            nbatch=nbatch,
            nstates=2,
            nvocab=256,
            out_wn=False,
            rnn_wn=True,
            rnn_type='mlstm',
            embd_wn=True,
        )
        global params
        params = [np.load('model/%d.npy'%i) for i in range(15)]
        params[2] = np.concatenate(params[2:6], axis=1)
        params[3:6] = []

        X = tf.placeholder(tf.int32, [None, hps.nsteps]) # input
        M = tf.placeholder(tf.float32, [None, hps.nsteps, 1]) # intermediate state
        S = tf.placeholder(tf.float32, [hps.nstates, None, hps.nhidden]) # internal and hidden states
        states = model(X, S, M, reuse=False)

        sess = tf.Session()
        tf.global_variables_initializer().run(session=sess)

        def seq_rep(xmb, mmb, smb):
            return sess.run(states, {X: xmb, M: mmb, S: smb})

        def transform(xs):
            tstart = time.time()
            xs = [preprocess(x) for x in xs]
            lens = np.asarray([len(x) for x in xs])
            sorted_idxs = np.argsort(lens)
            unsort_idxs = np.argsort(sorted_idxs)
            sorted_xs = [xs[i] for i in sorted_idxs] # here we sort input sentences by length, why?
            maxlen = np.max(lens)
            offset = 0
            n = len(xs) # number of input sentences
            smb = np.zeros((2, n, hps.nhidden), dtype=np.float32) # outputs
            # what is the second matrix smb[1, unsort_idxs, :]? it stores hidden states
            for step in range(0, ceil_round_step(maxlen, nsteps), nsteps):
                start = step
                end = step+nsteps
                xsubseq = [x[start:end] for x in sorted_xs] # pick first 64 characters in each sentence, why?
                ndone = sum([x == b'' for x in xsubseq])
                offset += ndone
                xsubseq = xsubseq[ndone:]
                sorted_xs = sorted_xs[ndone:]
                nsubseq = len(xsubseq)
                xmb, mmb = batch_pad(xsubseq, nsubseq, nsteps)
                for batch in range(0, nsubseq, nbatch):
                    start = batch
                    end = batch+nbatch
                    # if number of input sentences is smaller than "nbatch" xmb[start:end]=xmb
                    batch_smb = seq_rep(
                        xmb[start:end], mmb[start:end],
                        smb[:, offset+start:offset+end, :])
                    smb[:, offset+start:offset+end, :] = batch_smb
            features = smb[0, unsort_idxs, :]
            print('%0.3f seconds to transform %d examples' %
                  (time.time() - tstart, n))
            return features
        self.transform = transform

if __name__ == '__main__':

    mdl = Model()
    text = [
        'bad movie',
        'it was a great book',
        'actors were terrible',
        'it was ok',
        'worst movie ever!'
    ]
    text_features = mdl.transform(text)
    sentiment = text_features[:, 2388]
    print('sentiments')
    print(sentiment)

