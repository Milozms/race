import tensorflow as tf
import numpy as np
from func import stacked_gru, dot_attention, dropout
from tqdm import tqdm

class Model(object):
    def __init__(self, config, word_mat=None, char_mat=None, trainable=True, opt=True):
        N = config.batch_size*4
        self.article_maxlen, self.question_maxlen, self.opt_maxlen = config.para_limit, config.ques_limit, config.opt_limit
        self.config = config
        self.article_input = tf.placeholder(tf.int32, name='article', shape=[N, self.article_maxlen])
        self.question_input = tf.placeholder(tf.int32, name='question', shape=[N, self.question_maxlen])
        self.option_input = tf.placeholder(tf.int32, name='option', shape=[N, self.opt_maxlen])
        self.labels_input = tf.placeholder(tf.int32, name='label', shape=[N])
        self.global_step = tf.get_variable('global_step', shape=[], dtype=tf.int32, initializer=tf.constant_initializer(0), trainable=False)
        self.article = self.article_input
        self.question = self.question_input
        self.option = self.option_input
        self.labels = self.labels_input
        self.question = tf.concat([self.question, self.option], axis=1)
        # concat question and option
        self.emb_keep_prob = tf.get_variable("emb_keep_prob", shape=[
        ], dtype=tf.float32, trainable=False, initializer=tf.constant_initializer(config.emb_keep_prob))
        self.keep_prob = tf.get_variable("keep_prob", shape=[
        ], dtype=tf.float32, trainable=False, initializer=tf.constant_initializer(config.keep_prob))
        self.is_train = tf.get_variable("is_train", shape=[], dtype=tf.bool, trainable=False)
        self.word_mat = dropout(tf.get_variable("word_mat", initializer=tf.constant(
            word_mat, dtype=tf.float32), trainable=False), keep_prob=self.emb_keep_prob, is_train=self.is_train, mode="embedding")

        self.article_mask = tf.cast(self.article, tf.bool)
        self.question_mask = tf.cast(self.question, tf.bool)
        self.labels = tf.cast(self.labels, tf.float32)

        self.article_len = tf.reduce_sum(tf.cast(self.article_mask, tf.int32), axis=1)
        self.question_len = tf.reduce_sum(tf.cast(self.question_mask, tf.int32), axis=1)

        self.article_maxlen = tf.reduce_max(self.article_len)
        self.question_maxlen = tf.reduce_max(self.question_len)

        # self.article = tf.slice(self.article, [0, 0], [N, self.article_maxlen])
        # self.question = tf.slice(self.question, [0, 0], [N, self.question_maxlen])
        # self.article_mask = tf.slice(self.article_mask, [0, 0], [N, self.article_maxlen])
        # self.question_mask = tf.slice(self.question_mask, [0, 0], [N, self.question_maxlen])
        # self.labels = tf.slice(self.labels, [0], [N])
        self.define_model()

        if trainable:
            self.opt = tf.train.AdadeltaOptimizer(config.learning_rate)
            self.train_op = self.opt.minimize(self.loss)
            # self.lr = tf.get_variable("lr", shape=[], dtype=tf.float32, trainable=False)
            # self.opt = tf.train.AdadeltaOptimizer(learning_rate=self.lr, epsilon=1e-6)
            grads = self.opt.compute_gradients(self.loss)
            for grad, var in grads:
                tf.summary.histogram(var.name, var)
                tf.summary.histogram(var.name+'/gradient', grad)
            # gradients, variables = zip(*grads)
            # capped_grads, _ = tf.clip_by_global_norm(gradients, config.grad_clipping)
            # self.train_op = self.opt.apply_gradients(zip(capped_grads, variables), global_step=self.global_step)
        self.merged_summary_op = tf.summary.merge_all()

    def define_model_naive(self):
        config = self.config
        N, PL, QL, d = config.batch_size * 4, self.article_maxlen, self.question_maxlen, config.hidden_size

        with tf.variable_scope("emb"):
            with tf.name_scope("word"):
                c_emb = tf.nn.embedding_lookup(self.word_mat, self.article)
                q_emb = tf.nn.embedding_lookup(self.word_mat, self.question)

        with tf.variable_scope("encoding"):
            c, _ = stacked_gru(c_emb, d, batch=N, num_layers=3, seq_len=self.article_len, keep_prob=self.keep_prob,
                               is_train=self.is_train)
            tf.get_variable_scope().reuse_variables()
            q, q_state = stacked_gru(q_emb, d, batch=N, num_layers=3, seq_len=self.question_len, keep_prob=self.keep_prob,
                               is_train=self.is_train)
            # c size: [batch, c_len, 2*d]
            # q size: [batch, q_len, 2*d]
            q = tf.reduce_sum(q, axis=1)
            question_len_each = tf.tile(tf.expand_dims(self.question_len, axis=1), [1, 2*d])
            q = q / tf.cast(question_len_each, dtype=tf.float32)

        with tf.variable_scope("attention_q2d"):
            W_attention = tf.get_variable("W_attention", [2*d, 2*d], initializer=tf.truncated_normal_initializer(stddev=0.01))
            qw = tf.matmul(q, W_attention)
            # [batch, 2*d]
            qw = tf.expand_dims(qw, axis=2)
            # [batch, 2*d, 1]
            alpha = tf.matmul(c, qw)
            # [batch, clen, 1]
            alpha = alpha[:, :, 0]
            # [batch, clen]
            alpha = tf.nn.softmax(alpha, 1)
            alpha = tf.expand_dims(alpha, 1)
            # [batch, 1, clen]
            passage_vec = tf.matmul(alpha, c)
            # [batch, 1, 2d]
            passage_vec = tf.transpose(passage_vec, [0, 2, 1])
            # [batch, 2d, 1]

        with tf.variable_scope("predict"):
            p_hidden = 2*d
            q_hidden = 2*d
            W_predict = tf.get_variable("W_predict", [q_hidden, p_hidden], initializer=tf.truncated_normal_initializer(stddev=0.01))
            score = tf.matmul(q, W_predict)
            # [batch_size, p_hidden]
            score = tf.reshape(score, [-1, 1, p_hidden])
            # [batch_size, 1, p_hidden]
            score = tf.matmul(score, passage_vec)
            # [batch_size, 1, 1]
            score = score[:, 0, 0]
            self.score = tf.sigmoid(score)
            tf.summary.histogram('scores', self.score)
            self.loss = tf.losses.mean_squared_error(self.score, self.labels)
            tf.summary.scalar('loss_function', self.loss)

        self.debug_output_name = ['c', 'q', 'W1', 'W2', 'alpha', 'passage_vec', 'score']
        self.debug_output = [c, q, W_attention, alpha, passage_vec, W_predict, score]


    def define_model(self):
        config = self.config
        N, PL, QL, d = config.batch_size*4, self.article_maxlen, self.question_maxlen, config.hidden_size
        self.debug_output_name = []
        self.debug_output = []
        with tf.variable_scope("emb"):
            with tf.name_scope("word"):
                c_emb = tf.nn.embedding_lookup(self.word_mat, self.article)
                q_emb = tf.nn.embedding_lookup(self.word_mat, self.question)

        with tf.variable_scope("encoding"):
            c, _ = stacked_gru(c_emb, d, batch=N, num_layers=2, seq_len=self.article_len, keep_prob=self.keep_prob,
                               is_train=self.is_train)
            tf.get_variable_scope().reuse_variables()
            q, _ = stacked_gru(q_emb, d, batch=N, num_layers=2, seq_len=self.question_len, keep_prob=self.keep_prob,
                               is_train=self.is_train)
            # c size: [batch_size, c_len, 2*d]
            # q size: [batch_size, q_len, 2*d]

        with tf.variable_scope("attention_q2d"):
            qc_att, att_weight_ = dot_attention(c, q, mask=self.question_mask, hidden=d,
                                   keep_prob=self.keep_prob, is_train=self.is_train)
            # att_weight_ : [batch_size, c_len, q_len]
            # qc_att: [batch_size, c_len, 2*2*d]

            att, _ = stacked_gru(qc_att, d, num_layers=1, seq_len=self.article_len, batch=N,
                                 keep_prob=self.keep_prob, is_train=self.is_train)

        with tf.variable_scope("match"):
            self_att, self_att_weight_ = dot_attention(
                att, att, mask=self.article_mask, hidden=d, keep_prob=self.keep_prob, is_train=self.is_train)
            match, _ = stacked_gru(self_att, d, num_layers=1, seq_len=self.article_len, batch=N,
                                   keep_prob=self.keep_prob, is_train=self.is_train)
            # match size: [batch_size, c_len, 2*d]

        with tf.variable_scope("sum"):
            weight_for_each_passage_word = tf.expand_dims(tf.reduce_sum(att_weight_, 2), 1)
            # [batch_size, 1, c_len]
            passage_representation = tf.matmul(weight_for_each_passage_word, match)
            # [batch_size, 1, 2*d] -> [batch_size, 2*d]
            weight_for_each_question_word = tf.expand_dims(tf.reduce_sum(att_weight_, 1), 1)
            # [batch_size, 1, q_len]
            question_representation = tf.matmul(weight_for_each_question_word, q)
            # [batch_size, 1, 2*d] -> [batch_size, 2*d]

        with tf.variable_scope("predict"):
            p_hidden = 2*d
            q_hidden = 2*d
            W_predict = tf.get_variable("W_predict", [q_hidden, p_hidden], initializer=tf.truncated_normal_initializer(stddev=0.1), dtype=tf.float64)
            question_representation = tf.reshape(question_representation, [-1, q_hidden])
            # [batch_size, q_hidden]
            question_representation = tf.cast(question_representation, dtype=tf.float64)
            score = tf.matmul(question_representation, W_predict)
            # [batch_size, p_hidden]
            score = tf.reshape(score, [-1, 1, p_hidden])
            # [batch_size, 1, p_hidden]
            passage_representation = tf.transpose(passage_representation, [0, 2, 1])
            passage_representation = tf.cast(passage_representation, dtype=tf.float64)
            score = tf.matmul(score, passage_representation)
            # [batch_size, 1, 1]
            score = tf.reshape(score, [-1, 4])
            score = tf.nn.softmax(score, dim=1)
            score = tf.cast(score, dtype=tf.float32)
            self.score = tf.reshape(score, [-1])
            tf.summary.histogram('scores', self.score)
            self.loss = tf.losses.mean_squared_error(self.score, self.labels)
            tf.summary.scalar('loss_function', self.loss)
        self.debug_output_name = ['att_weight_', 'score']
        self.debug_output = [att_weight_, self.score]


    def get_score(self):
        return self.score

    def get_loss(self):
        return self.loss

    def get_global_step(self):
        return self.global_step
