#! /usr/bin/env python

import tensorflow as tf
import inspect
import numpy as np
import os, sys
sys.path.append("..")
import os.path
from singleton import Singleton
import codecs
import heapq
from sys import argv

import rnn_text_classification.vector_helper as wv

import rnn_text_classification.tf_flags

FLAGS = tf.flags.FLAGS
FLAGS._parse_flags()

class Classify_CN(metaclass=Singleton):
    def __init__(self, module_dir):
        print("module dir: " + module_dir)

        checkpoint_dir = os.path.join(module_dir, "cn", "checkpoints")
        print('checkpoint_dir:' + checkpoint_dir)
        classes_file = codecs.open(os.path.join(module_dir, "cn", "classes"), "r", "utf-8")
        self.classes = list(line.strip() for line in classes_file.readlines())
        classes_file.close()

        # Evaluation
        # ==================================================

        def lstm_cell():
            if 'reuse' in inspect.signature(tf.contrib.rnn.BasicLSTMCell.__init__).parameters:
                return tf.contrib.rnn.BasicLSTMCell(FLAGS.hidden_neural_size, forget_bias=0.0,
                                                    state_is_tuple=True,
                                                    reuse=tf.get_variable_scope().reuse)
            else:
                return tf.contrib.rnn.BasicLSTMCell(
                    FLAGS.hidden_layer_num, forget_bias=0.0, state_is_tuple=True)

        attn_cell = lstm_cell

        cell = tf.contrib.rnn.MultiRNNCell(
            [attn_cell() for _ in range(FLAGS.hidden_layer_num)], state_is_tuple=True)

        self.batch_size = 1
        self.sentence_words_num = FLAGS.max_len
        self.embedding_dim_cn = FLAGS.emdedding_dim

        checkpoint_file = tf.train.latest_checkpoint(checkpoint_dir)
        print('checkpoint_file: ' + checkpoint_file)
        graph = tf.Graph()
        with graph.as_default():
            with tf.device("/cpu:0"):
                session_conf = tf.ConfigProto(
                    allow_soft_placement=FLAGS.allow_soft_placement,
                    log_device_placement=FLAGS.log_device_placement)
                session_conf.gpu_options.allow_growth = True
                self.session = tf.Session(config=session_conf)

                self._initial_state = cell.zero_state(self.batch_size, dtype=tf.float32)

                # Load the saved meta graph and restore variables
                saver = tf.train.import_meta_graph("{}.meta".format(checkpoint_file))
                saver.restore(self.session, checkpoint_file)

                # for op in graph.get_operations():
                #     print(op.name)

                # Get the placeholders from the graph by name
                self.embedded_chars = graph.get_operation_by_name("model_1/embedded_chars").outputs[0]
                self.mask_x = graph.get_operation_by_name("model_1/mask_x").outputs[0]
                # self.dropout_keep_prob = graph.get_operation_by_name("dropout_keep_prob").outputs[0]

                # Tensors we want to evaluate
                self.prediction = graph.get_operation_by_name("model_1/accuracy/prediction").outputs[0]
                self.scores = graph.get_operation_by_name("model_1/Softmax_layer_and_output/scores").outputs[0]
                self.probability = graph.get_operation_by_name("model_1/accuracy/probability").outputs[0]

    def __enter__(self):
        print('Classify_CN enter')

    def __exit__(self):
        print('Classify_CN exit')
        self.session.close()

    def getCategory(self, sentence, reverse=True):
        # start_time = time.time()
        if reverse==True:
            sen_list = sentence.strip().split(' ')
            sen_list.reverse()
            sentence = ' '.join(sen_list)
        else:
            pass

        x_test = [sentence]

        def generate_mask(set_x):
            masked_x = np.zeros([self.sentence_words_num, len(set_x)])
            for i, x in enumerate(set_x):
                x_list = x.split(' ')
                if len(x_list) < self.sentence_words_num:
                    masked_x[0:len(x_list), i] = 1
                else:
                    masked_x[:, i] = 1
            return masked_x

        mask_x = generate_mask(x_test)

        sentence_embedded_chars = wv.embedding_lookup(len(x_test), self.sentence_words_num,
                                                      self.embedding_dim_cn, x_test)

        data_embed = (sentence_embedded_chars, mask_x)

        # for step, (x, y, mask_x) in enumerate(data_helper.batch_iter(data_embed, batch_size=self.batch_size)):
        # fetches = [self.prediction, self.scores, self.probability]
        feed_dict = {}
        feed_dict[self.embedded_chars] = data_embed[0]
        feed_dict[self.mask_x] = data_embed[1]

        state = self.session.run(self._initial_state)
        for i, (c, h) in enumerate(self._initial_state):
           feed_dict[c] = state[i].c
           feed_dict[h] = state[i].h

        # [predictions, scores, probabilities] = self.session.run(fetches, feed_dict)
        predictions = self.session.run(self.prediction, feed_dict)
        probabilities = self.session.run(self.probability, feed_dict)
        scores = self.session.run(self.scores, feed_dict)
        # print('scores: ', scores[0])
        # print('probabilities: ', probabilities[0])
        # print('predictions:', predictions)
        # if self.classes >= 5:

        prediction_index = predictions[0]
        max_score = scores[0][prediction_index]
        class_prediction = self.classes[prediction_index]
        max_probability = probabilities[0][prediction_index]

        # can_score = scores[0][top5_index[1]]
        # can_class = self.classes[top5_index[1]]

        result = {}
        result['score'] = float(max_score)
        result['probability'] = float(max_probability)
        result['value'] = class_prediction.strip()
        if len(self.classes) >= 5:
            top5_index = heapq.nlargest(5, range(len(scores[0])), scores[0].take)
            top5_class = []
            top5_score = []
            top5_probability = []
            for i in range(5):
                top5_score.append(float(scores[0][top5_index[i]]))
                top5_class.append(self.classes[top5_index[i]])
                top5_probability.append(float(probabilities[0][top5_index[i]]))
            result['top5_value'] = top5_class
            result['top5_score'] = top5_score
            result['top5_probability'] = top5_probability

        # end_time = time.time()
        #
        # time_cost = end_time-start_time
        # return (result, time_cost)
        return result

        # prediction_index = predictions[0]
        # max_score = scores[0][prediction_index]
        # class_prediction = self.classes[prediction_index]
        # max_probability = probabilities[0][prediction_index]
        #
        # scores[0][prediction_index] = -1
        # probabilities[0][prediction_index] = -1
        # can_score = max(scores[0])
        # can_probability = max(probabilities[0])
        # can_index = list(scores[0]).index(can_score)
        # can_class = self.classes[can_index]
        #
        # result = {}
        # result['score'] = max_score
        # result['probability'] = max_probability
        # result['value'] = class_prediction
        # result['score_can'] = can_score
        # result['value_can'] = can_class
        # result['probability_can'] = can_probability

        # end_time = time.time()
        #
        # time_cost = end_time-start_time
        # return (result, time_cost)
        # return result

if __name__ == '__main__':
    # script, module_path = argv
    module_path = './runs/people2014'
    classify = Classify_CN(module_path)
    # from tools.word_cut import WordCutHelper

    # wh = WordCutHelper(0)

    while 1:
        sentence = input('sentence: ')
        if not sentence:
            break
        # result, time_cost = classify.getCategory(sentence)

        # value = wh.getWords(sentence)
        # sentence = ' '.join(value)
        print(sentence)

        result = classify.getCategory(sentence)
        print(result)
        print(result['value'])
        print('\n')
        # print('Time used: {} s'.format(time_cost))
