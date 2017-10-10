# coding:utf-8

import sys
sys.path.append("..")
import tensorflow as tf
import os
import time
from rnn_text_classification.rnn_model import RNN_Model
from rnn_text_classification import data_helper
import codecs
import rnn_text_classification.vector_helper as wv
from sys import argv
import rnn_text_classification.tf_flags
import re

dataset_path = '../work_space/people2014/dataset/words'
dataset_name = 'people2014'
# script, dataset_path, dataset_name = argv

tf.flags.DEFINE_string('out_dir',
                       os.path.abspath(os.path.join(os.path.curdir, "runs", dataset_name, "cn")), 'output directory')
# add
# tf.flags.DEFINE_float("test_sample_percentage", .02, "Percentage of the data to use for test")
tf.flags.DEFINE_float("valid_sample_percentage", .002, "Percentage of the data to use for valid")

FLAGS = tf.flags.FLAGS
FLAGS._parse_flags()
print("\nParameters:")
for attr, value in sorted(FLAGS.__flags.items()):
    print("{}={}".format(attr.upper(), value))
print("")

class Config(object):

    hidden_neural_size = FLAGS.hidden_neural_size
    embed_dim = FLAGS.emdedding_dim
    hidden_layer_num = FLAGS.hidden_layer_num
    keep_prob = FLAGS.keep_prob
    lr = FLAGS.lr
    lr_decay = FLAGS.lr_decay
    batch_size = FLAGS.batch_size
    num_step = FLAGS.max_len
    max_grad_norm = FLAGS.max_grad_norm
    num_epoch = FLAGS.num_epoch
    max_decay_epoch = FLAGS.max_decay_epoch
    valid_num = FLAGS.valid_num
    out_dir = FLAGS.out_dir
    checkpoint_every = FLAGS.check_point_every
    # test_sample_percentage = FLAGS.test_sample_percentage
    valid_portion = FLAGS.valid_sample_percentage
    allow_soft_placement = FLAGS.allow_soft_placement
    log_device_placement = FLAGS.log_device_placement
    init_scale = FLAGS.init_scale

def evaluate(config, model,session,data,global_steps=None,summary_writer=None):

    correct_num=0
    total_num=len(data[0])
    for step, (x, y, mask_x) in enumerate(data_helper.batch_iter(data,batch_size=config.batch_size)):

        x_embedded = wv.embedding_lookup(len(list(x)), config.num_step, config.embed_dim, list(x), 1)

        fetches = model.correct_num
        feed_dict={}
        feed_dict[model.embedded_x] = x_embedded
        feed_dict[model.target] = y
        feed_dict[model.mask_x] = mask_x

        state = session.run(model._initial_state)
        for i, (c, h) in enumerate(model._initial_state):
            feed_dict[c] = state[i].c
            feed_dict[h] = state[i].h

        count = session.run(fetches, feed_dict)
        correct_num += count

    accuracy = float(correct_num)/total_num
    dev_summary = tf.summary.scalar('dev_accuracy', accuracy)
    dev_summary = session.run(dev_summary)
    if summary_writer:
        summary_writer.add_summary(dev_summary, global_steps)
        summary_writer.flush()
    return accuracy

def run_epoch(config, eval_config,
              model, session, data, global_steps,
              valid_model, valid_data,
              train_summary_writer=None, valid_summary_writer=None):

    for step, (x, y, mask_x) in enumerate(data_helper.batch_iter(data, batch_size=config.batch_size)):
        feed_dict = {}
        # word embedding
        x_embedded = wv.embedding_lookup(len(list(x)), config.num_step, config.embed_dim, list(x), 0)
        feed_dict[model.embedded_x] = x_embedded
        feed_dict[model.target] = y
        feed_dict[model.mask_x] = mask_x
        # model.assign_new_batch_size(session,len(x))
        fetches = [model.cost, model.accuracy, model.train_op,model.summary]
        state = session.run(model._initial_state)
        for i, (c,h) in enumerate(model._initial_state):
            feed_dict[c] = state[i].c
            feed_dict[h] = state[i].h
        cost,accuracy,_,summary = session.run(fetches,feed_dict)
        if train_summary_writer:
            train_summary_writer.add_summary(summary,global_steps)
            train_summary_writer.flush()
        print("the %i step, train cost is: %f and the train accuracy is %f" %
              (global_steps, cost, accuracy))
        if global_steps % 1000 == 0:
            valid_accuracy = evaluate(eval_config, valid_model, session, valid_data, global_steps, valid_summary_writer)
            print("the %i step, train cost is: %f and the train accuracy is %f and the valid accuracy is %f\n" %
                  (global_steps, cost, accuracy, valid_accuracy))
        global_steps += 1
        del x_embedded

    return global_steps

def train_step(data_enhance=False, data_reverse=False):

    # Load data
    print("Loading data...")

    config = Config()
    eval_config=Config()
    eval_config.keep_prob=1.0
    eval_config.batch_size = 1

    classify_files = []
    classify_names = []
    for parent, dirnames, filenames in os.walk(dataset_path):
        for filename in filenames:
            # 这里是为了避免存在缓存文件
            if filename[-1] == '~':
                # os.remove(os.path.join(dataset_path, filename))
                continue
            else:
                classify_files.append(os.path.join(dataset_path, filename))
                classify_names.append(filename)

    # load_data(classify_files, config, sort_by_len=True, enhance = True, reverse=True)
    train_data, valid_data = data_helper.load_data(classify_files, config, sort_by_len=True,
                                                   enhance=data_enhance, reverse=data_reverse)
    num_classes = len(train_data[1][0])

    print("begin training")

    # gpu_config=tf.ConfigProto()
    # gpu_config.gpu_options.allow_growth=True
    with tf.Graph().as_default():
        session_conf = tf.ConfigProto(
            allow_soft_placement=config.allow_soft_placement,
            log_device_placement=config.log_device_placement)
        session = tf.Session(config=session_conf)

        initializer = tf.random_uniform_initializer(-1*FLAGS.init_scale,1*FLAGS.init_scale)
        with tf.variable_scope("model", reuse=None,initializer=initializer):
            model = RNN_Model(config=config, num_classes=num_classes, is_training=True)

        with tf.variable_scope("model", reuse=True,initializer=initializer):
            valid_model = RNN_Model(config=eval_config,num_classes=num_classes, is_training=False)
            # test_model = RNN_Model(config=eval_config,num_classes=num_classes,is_training=False)

        # # add summary
        # train_summary_dir = os.path.join(config.out_dir,"summaries","train")
        # train_summary_writer = tf.summary.FileWriter(train_summary_dir,session.graph)
        #
        # dev_summary_dir = os.path.join(eval_config.out_dir,"summaries","dev")
        # dev_summary_writer = tf.summary.FileWriter(dev_summary_dir, session.graph)

        # add checkpoint
        checkpoint_dir = os.path.abspath(os.path.join(config.out_dir, "checkpoints"))
        checkpoint_prefix = os.path.join(checkpoint_dir, "model")
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        saver = tf.train.Saver(tf.global_variables(), max_to_keep=FLAGS.check_point_every)

        # Write classify names
        classes_file = codecs.open(os.path.join(config.out_dir, "classes"), "w", "utf-8")
        for classify_name in classify_names:
            classes_file.write(classify_name)
            classes_file.write('\n')
        classes_file.close()

        session.run(tf.global_variables_initializer())
        global_steps = 1
        begin_time = int(time.time())

        for i in range(config.num_epoch):
            print("the %d epoch training..." % (i+1))
            lr_decay = config.lr_decay ** max(i-config.max_decay_epoch, 0)
            model.assign_new_lr(session, config.lr*lr_decay)

            # global_steps = run_epoch(
            #     config, eval_config,
            #     model, session, train_data,
            #     global_steps, valid_model,
            #     valid_data,
            #     train_summary_writer, dev_summary_writer)
            global_steps = run_epoch(
                config, eval_config,
                model, session, train_data,
                global_steps, valid_model,
                valid_data)

            if i % config.checkpoint_every == 0:
                path = saver.save(session, checkpoint_prefix, global_steps)
                print("Saved model checkpoint to{}\n".format(path))

        print("the train is finished")
        end_time = int(time.time())
        print("training takes %d seconds already\n" % (end_time-begin_time))
        # test_accuracy = evaluate(eval_config, test_model, session, test_data)
        # print("the test data accuracy is %f" % test_accuracy)
        print("program end!")

        # 修改checkpoint文件中的model路径
        lines = []
        with open(os.path.join(checkpoint_dir, "checkpoint"), "r") as f:
            for line in f.readlines():
                line_deal = replace_model_path(line)
                lines.append(line_deal)

        with open(os.path.join(checkpoint_dir, "checkpoint"), "w") as f:
            for line in lines:
                f.write(line)

def replace_model_path(checkpoint_path):
    pat = re.compile(r'\"(/.+/)model-\d+\"$')
    model = ''.join(pat.findall(checkpoint_path))
    text = re.sub(model, '', checkpoint_path)
    return text

if __name__ == "__main__":
    train_step(data_enhance=False)
    print('done')






