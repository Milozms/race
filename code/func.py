import tensorflow as tf
from tensorflow.python.ops.rnn_cell_impl import _linear, RNNCell

INF = 1e30

def stacked_gru(inputs, hidden, num_layers, seq_len, batch, keep_prob=1.0, is_train=None, std=0.1, concat_layers=True, scope="StackedGRU"):
    m_cell_fw = []
    m_cell_bw = []
    for i in range(num_layers):
        cell_fw = tf.nn.rnn_cell.GRUCell(num_units=hidden,
                                         kernel_initializer=tf.truncated_normal_initializer(stddev=std))
        cell_fw = tf.nn.rnn_cell.DropoutWrapper(cell=cell_fw, input_keep_prob=1.0, output_keep_prob=keep_prob)
        cell_bw = tf.nn.rnn_cell.GRUCell(num_units=hidden,
                                         kernel_initializer=tf.truncated_normal_initializer(stddev=std))
        cell_bw = tf.nn.rnn_cell.DropoutWrapper(cell=cell_bw, input_keep_prob=1.0, output_keep_prob=keep_prob)
        m_cell_fw.append(cell_fw)
        m_cell_bw.append(cell_bw)

    m_cell_fw = tf.nn.rnn_cell.MultiRNNCell(m_cell_fw, state_is_tuple=True)
    m_cell_bw = tf.nn.rnn_cell.MultiRNNCell(m_cell_bw, state_is_tuple=True)
    init_state_fw = m_cell_fw.zero_state(batch, dtype=tf.float32)
    init_state_bw = m_cell_bw.zero_state(batch, dtype=tf.float32)
    output, state = tf.nn.bidirectional_dynamic_rnn(cell_fw=m_cell_fw, cell_bw=m_cell_bw, inputs=inputs,
                                                    sequence_length=seq_len, scope=scope, dtype=tf.float32)
    output = tf.concat([output[0], output[1]], axis=2)
    return output, state


'''
def stacked_gru(inputs, hidden, num_layers, seq_len, batch=None, keep_prob=1.0, is_train=None, std=0.01, concat_layers=True, scope="StackedGRU"):
    with tf.variable_scope(scope):
        outputs = [inputs]
        for layer in range(num_layers):
            with tf.variable_scope("Layer_{}".format(layer)):
                with tf.variable_scope("fw"):
                    inputs_fw = dropout(
                        outputs[-1], keep_prob=keep_prob, is_train=is_train)
                    cell_fw = GRUCell(hidden)
                    init_fw = None
                    if batch is not None:
                        init_fw = tf.tile(tf.get_variable("init_fw", [
                                          1, hidden], initializer=tf.truncated_normal_initializer(stddev=std)), [batch, 1])
                    out_fw, state_fw = tf.nn.dynamic_rnn(
                        cell_fw, inputs_fw, sequence_length=seq_len, initial_state=init_fw, dtype=tf.float32)
                with tf.variable_scope("bw"):
                    _inputs_bw = tf.reverse_sequence(
                        outputs[-1], seq_lengths=seq_len, seq_dim=1, batch_dim=0)
                    inputs_bw = dropout(
                        _inputs_bw, keep_prob=keep_prob, is_train=is_train)
                    cell_bw = GRUCell(hidden)
                    init_bw = None
                    if batch is not None:
                        init_bw = tf.tile(tf.get_variable("init_bw", [
                                          1, hidden], initializer=tf.truncated_normal_initializer(stddev=std)), [batch, 1])
                    out_bw, state_bw = tf.nn.dynamic_rnn(
                        cell_bw, inputs_bw, sequence_length=seq_len, initial_state=init_bw, dtype=tf.float32)

                    out_bw = tf.reverse_sequence(
                        out_bw, seq_lengths=seq_len, seq_dim=1, batch_dim=0)
                    # out_bw size: [batch_size, seq_len, hidden_dim]
                outputs.append(tf.concat([out_fw, out_bw], axis=2))
        # x in outputs size: [batch_size, seq_len, 2*hidden_dim]
        if concat_layers:
            res = tf.concat(outputs[1:], axis=2)
            # res size: [batch_size, seq_len, 2*num_layers*hidden_dim]
        else:
            res = outputs[-1]
            # res = tf.reduce_mean(outputs[1:], axis=2)
        res_mean = tf.reduce_mean(outputs[1:], axis=0)
        state = tf.concat([state_fw, state_bw], axis=1)
        return res, res_mean, state
'''


def dropout(args, keep_prob, is_train, mode="recurrent"):
    noise_shape = None
    shape = tf.shape(args)
    if mode == "embedding":
        noise_shape = [shape[0], 1]
    if mode == "recurrent" and len(args.get_shape().as_list()) == 3:
        noise_shape = [shape[0], 1, shape[-1]]
    args = tf.cond(is_train, lambda: tf.nn.dropout(
        args, keep_prob, noise_shape=noise_shape), lambda: args)
    return args


def softmax_mask(val, mask):
    # val and mask have same shape
    # 0 -> -INF
    # 1 -> val
    return -INF * (1 - tf.cast(mask, tf.float32)) + val

def softmax2d(logits):
    sum = tf.reduce_sum(tf.exp(logits), [1, 2])
    sum = tf.expand_dims(sum, axis=1)
    sum = tf.expand_dims(sum, axis=2)
    ilen = tf.shape(logits)[1]
    mlen = tf.shape(logits)[2]
    sum = tf.tile(sum, [1, ilen, mlen])
    return tf.exp(logits) / sum


def dot_attention(inputs, memory, mask, hidden, keep_prob=1.0, is_train=None, scope="dot_attention"):
    with tf.variable_scope(scope):

        d_inputs = inputs
        d_memory = memory

        JX = tf.shape(inputs)[1]
        # ilen
        inputs_ = tf.nn.relu(tf.layers.dense(d_inputs, hidden, name="inputs"))
        memory_ = tf.nn.relu(tf.layers.dense(d_memory, hidden, name="memory"))
        # [batch, ilen, hidden]
        # [batch, mlen, hidden]

        outputs = tf.matmul(inputs_, tf.transpose(
            memory_, [0, 2, 1])) / (hidden ** 0.5)
        # the s in paper
        # [batch, ilen, mlen]
        mask = tf.tile(tf.expand_dims(mask, axis=1), [1, JX, 1])
        # tf.expand_dims(mask, axis=1): [batch, 1, mlen]
        # mask: [batch, ilen, mlen]
        logits = tf.nn.softmax(softmax_mask(outputs, mask))
        # the a in paper
        # [batch, ilen, mlen]
        att_weight = softmax2d(softmax_mask(outputs, mask))
        # attention weight for each question word
        # [batch, ilen, mlen]
        outputs = tf.matmul(logits, memory)
        # the c in paper
        # [batch, ilen, memory_dim]
        res = tf.concat([inputs, outputs], axis=2)
        # [batch, ilen, inputs_dim + memory_dim]

        dim = res.get_shape().as_list()[-1]
        gate = tf.nn.sigmoid(tf.layers.dense(res, dim, use_bias=False, name="gate"))
        return res * gate, att_weight

'''
def dot_attention(inputs, memory, mask, hidden, keep_prob=1.0, is_train=None, scope="dot_attention"):
    with tf.variable_scope(scope):

        d_inputs = dropout(inputs, keep_prob=keep_prob, is_train=is_train)
        d_memory = dropout(memory, keep_prob=keep_prob, is_train=is_train)

        JX = tf.shape(inputs)[1]
        # ilen
        inputs_ = tf.nn.relu(dense(d_inputs, hidden, scope="inputs"))
        memory_ = tf.nn.relu(dense(d_memory, hidden, scope="memory"))
        # [batch, ilen, hidden]
        # [batch, mlen, hidden]

        outputs = tf.matmul(inputs_, tf.transpose(
            memory_, [0, 2, 1])) / (hidden ** 0.5)
        # the s in paper
        # [batch, ilen, mlen]
        mask = tf.tile(tf.expand_dims(mask, axis=1), [1, JX, 1])
        # tf.expand_dims(mask, axis=1): [batch, 1, mlen]
        # mask: [batch, ilen, mlen]
        logits = tf.nn.softmax(softmax_mask(outputs, mask))
        # the a in paper
        # [batch, ilen, mlen]
        att_weight = softmax2d(softmax_mask(outputs, mask))
        # attention weight for each question word
        # [batch, ilen, mlen]
        outputs = tf.matmul(logits, memory)
        # the c in paper
        # [batch, ilen, memory_dim]
        res = tf.concat([inputs, outputs], axis=2)
        # [batch, ilen, inputs_dim + memory_dim]

        dim = res.get_shape().as_list()[-1]
        gate = tf.nn.sigmoid(dense(res, dim, use_bias=False, scope="gate"))
        return res * gate, att_weight
'''
'''
def dense(inputs, hidden, use_bias=True, scope="dense", std=0.1):
    with tf.variable_scope(scope):
        shape = tf.shape(inputs)                      # shape of inputs: [batch, len, dim]
        dim = inputs.get_shape().as_list()[-1]        # dim of input
        out_shape = [shape[idx] for idx in range(
            len(inputs.get_shape().as_list()) - 1)] + [hidden]  # [batch, len, dim, hidden]
        flat_inputs = tf.reshape(inputs, [-1, dim])
        W = tf.get_variable("W", [dim, hidden], initializer=tf.truncated_normal_initializer(stddev=std))
        res = tf.matmul(flat_inputs, W)
        if use_bias:
            b = tf.get_variable(
                "b", [hidden], initializer=tf.constant_initializer(0.))
            res = tf.nn.bias_add(res, b)
        res = tf.reshape(res, out_shape)
        return res
'''

'''
class GRUCell(RNNCell):

    def __init__(self, num_units, reuse=None):
        super(GRUCell, self).__init__(_reuse=reuse)
        self._num_units = num_units
        self._gate_linear = None
        self._candidate_linear = None

    @property
    def state_size(self):
        return self._num_units

    @property
    def output_size(self):
        return self._num_units

    def call(self, inputs, state):
        if self._gate_linear is None:
            with tf.variable_scope("gates"):
                self._gate_linear = _linear([inputs, state], 2 * self._num_units, True,
                                            kernel_initializer=tf.orthogonal_initializer(1.0), bias_initializer=tf.constant_initializer(1.0))
        value = tf.sigmoid(self._gate_linear([inputs, state]))
        r, u = tf.split(value=value, num_or_size_splits=2, axis=1)

        r_state = r * state
        if self._candidate_linear is None:
            with tf.variable_scope("candidate"):
                self._candidate_linear = _linear([inputs, r_state], self._num_units, True, kernel_initializer=tf.orthogonal_initializer(
                    1.0), bias_initializer=tf.constant_initializer(-1.0))
        c = tf.nn.tanh(self._candidate_linear([inputs, r_state]))
        new_h = u * state + (1 - u) * c
        return new_h, new_h
'''