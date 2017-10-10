import tensorflow as tf
import os

# Parameters
# ==================================================

tf.flags.DEFINE_integer('batch_size', 32, 'the batch_size of the training procedure')
tf.flags.DEFINE_float('lr', 0.1, 'the learning rate')
tf.flags.DEFINE_float('lr_decay', 0.5, 'the learning rate decay')
tf.flags.DEFINE_integer('valid_num', 100, 'epoch num of validation')
tf.flags.DEFINE_integer('checkpoint_num', 1000, 'epoch num of checkpoint')
tf.flags.DEFINE_float('init_scale', 0.1, 'init scale')
tf.flags.DEFINE_float('keep_prob', 0.5, 'dropout rate')
tf.flags.DEFINE_integer('num_epoch', 60, 'num epoch')
tf.flags.DEFINE_integer('max_decay_epoch', 30, 'max epoch of decay')
tf.flags.DEFINE_integer('max_grad_norm', 5, 'max_grad_norm')
tf.flags.DEFINE_integer('check_point_every', 10, 'checkpoint every num epoch ')

# Model Hyperparameters
tf.flags.DEFINE_integer('emdedding_dim', 300, 'embedding dim')
tf.flags.DEFINE_integer('hidden_neural_size', 1000, 'LSTM hidden neural size')
tf.flags.DEFINE_integer('hidden_layer_num', 1, 'LSTM hidden layer num')
tf.flags.DEFINE_integer('max_len', 30, 'max_len of training sentence')

# Misc Parameters
tf.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
tf.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")
