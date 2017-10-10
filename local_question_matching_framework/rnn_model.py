import inspect
import tensorflow as tf

class RNN_Model(object):

    def __init__(self, config, num_classes, is_training=True):

        keep_prob = config.keep_prob
        batch_size = config.batch_size

        num_step = config.num_step
        embed_dim = config.embed_dim
        self.embedded_x = tf.placeholder(tf.float32, [None, num_step, embed_dim], name="embedded_chars")
        self.target = tf.placeholder(tf.int64, [None, num_classes], name='target')
        self.mask_x = tf.placeholder(tf.float32, [num_step, None], name="mask_x")

        hidden_neural_size=config.hidden_neural_size
        hidden_layer_num=config.hidden_layer_num

        # build LSTM network
        def lstm_cell():
            if 'reuse' in inspect.signature(tf.contrib.rnn.BasicLSTMCell.__init__).parameters:
                return tf.contrib.rnn.BasicLSTMCell(hidden_neural_size, forget_bias=0.0,
                                                    state_is_tuple=True,
                                                    reuse=tf.get_variable_scope().reuse)
            else:
                return tf.contrib.rnn.BasicLSTMCell(
                    hidden_neural_size, forget_bias=0.0, state_is_tuple=True)

        attn_cell = lstm_cell

        if is_training and keep_prob < 1:
            def attn_cell():
                return tf.contrib.rnn.DropoutWrapper(
                    lstm_cell(), output_keep_prob=config.keep_prob)

        cell = tf.contrib.rnn.MultiRNNCell(
            [attn_cell() for _ in range(hidden_layer_num)], state_is_tuple=True)

        self._initial_state = cell.zero_state(batch_size, dtype=tf.float32)

        inputs = self.embedded_x

        if keep_prob < 1:
            inputs = tf.nn.dropout(inputs, keep_prob)

        out_put = []
        state = self._initial_state
        with tf.variable_scope("LSTM_layer"):
            for time_step in range(num_step):
                if time_step > 0: tf.get_variable_scope().reuse_variables()
                (cell_output, state) = cell(inputs[:, time_step,:],state)
                out_put.append(cell_output)

        out_put=out_put*self.mask_x[:,:,None]

        with tf.name_scope("mean_pooling_layer"):
            out_put = tf.reduce_sum(out_put,0)/(tf.reduce_sum(self.mask_x,0)[:,None])

        with tf.name_scope("Softmax_layer_and_output"):
            softmax_w = tf.get_variable("softmax_w",[hidden_neural_size,num_classes],dtype=tf.float32)
            softmax_b = tf.get_variable("softmax_b",[num_classes],dtype=tf.float32)
            # self.logits = tf.matmul(out_put,softmax_w)
            # self.scores = tf.add(self.logits, softmax_b, name='scores')
            self.scores = tf.nn.xw_plus_b(out_put, softmax_w, softmax_b, name="scores")

        with tf.name_scope("loss"):
            self.loss = tf.nn.softmax_cross_entropy_with_logits(labels=self.target, logits=self.scores + 1e-10)
            self.cost = tf.reduce_mean(self.loss)

        with tf.name_scope("accuracy"):
            self.prediction = tf.argmax(self.scores, 1, name="prediction")
            correct_prediction = tf.equal(self.prediction, tf.argmax(self.target, 1))
            self.correct_num = tf.reduce_sum(tf.cast(correct_prediction, tf.float32))
            self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32), name="accuracy")
            self.probability = tf.nn.softmax(self.scores, name="probability")

        # add summary
        loss_summary = tf.summary.scalar("loss", self.cost)
        # add summary
        accuracy_summary = tf.summary.scalar("accuracy_summary", self.accuracy)

        if not is_training:
            return

        self.global_step = tf.Variable(0, name="global_step", trainable=False)
        self.lr = tf.Variable(0.0, trainable=False)

        tvars = tf.trainable_variables()
        grads, _ = tf.clip_by_global_norm(tf.gradients(self.cost, tvars), config.max_grad_norm)

        # Keep track of gradient values and sparsity (optional)
        grad_summaries = []
        for g, v in zip(grads, tvars):
            if g is not None:
                grad_hist_summary = tf.summary.histogram("{}/grad/hist".format(v.name), g)
                sparsity_summary = tf.summary.scalar("{}/grad/sparsity".format(v.name), tf.nn.zero_fraction(g))
                grad_summaries.append(grad_hist_summary)
                grad_summaries.append(sparsity_summary)
        self.grad_summaries_merged = tf.summary.merge(grad_summaries)

        self.summary = tf.summary.merge([loss_summary,accuracy_summary,self.grad_summaries_merged])

        optimizer = tf.train.GradientDescentOptimizer(self.lr)
        optimizer.apply_gradients(zip(grads, tvars))
        self.train_op=optimizer.apply_gradients(zip(grads, tvars))

        self.new_lr = tf.placeholder(tf.float32,shape=[],name="new_learning_rate")
        self._lr_update = tf.assign(self.lr,self.new_lr)

    def assign_new_lr(self,session,lr_value):
        session.run(self._lr_update,feed_dict={self.new_lr:lr_value})

    # add
    def predict_label(self, sess, data_embed, classes):
        # x = np.array(text)
        # feed = {self.embedded_x: x,

        # }
        fetches = self.prediction
        feed_dict = {}
        feed_dict[self.embedded_x] = data_embed[0]
        feed_dict[self.mask_x] = data_embed[1]

        state = sess.run(self._initial_state)
        for i, (c,h) in enumerate(self._initial_state):
           feed_dict[c]=state[i].c
           feed_dict[h]=state[i].h

        y_pred = sess.run(fetches, feed_dict)
        prediction_index = y_pred[0]
        class_prediction = classes[prediction_index]
        result = {}
        result['value'] = class_prediction
        return result

        # probs, state = sess.run([self.probs, self.final_state], feed_dict=feed)

        # results = np.argmax(probs, 1)
        # id2labels = dict(zip(labels.values(), labels.keys()))
        # labels = map(id2labels.get, results)
        # return labels