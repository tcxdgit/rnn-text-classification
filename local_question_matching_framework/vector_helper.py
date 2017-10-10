import sys,os
sys.path.append("..")
import config as cf
from benebot_vector.vector import BenebotVector
import numpy as np

loop_word = 0
dim_shrink = 1
sentence_vector_dict = {}

bv = BenebotVector(cf.getPathConfig('v'))
# print('done')

def getVector(word, embedding_dim):
    vector = bv.getVectorByWord(word)
    #print(vector)
    if vector:
        return vector
    return [0.0] * embedding_dim

def embedding_lookup(sequence_num, sequence_length, embedding_dim, input_x, maintain = 0):
    for sentence_index in range(sequence_num):
        sentence_str = input_x[sentence_index]
        if not (sentence_str in sentence_vector_dict):
            #print(sentence_str)
            sentence = sentence_str.split(" ")
            sentence_length = len(sentence)
            loop_count = loop_word
            if loop_count > sentence_length:
                loop_count = sentence_length
            for word_index in range(sequence_length):
                if word_index >= sentence_length:
                    loop_index = word_index - sentence_length
                    if loop_index < loop_count:
                        word_vector_tmp = np.array([getVector(sentence[loop_index], embedding_dim)*dim_shrink])
                    else:
                        word_vector_tmp = np.array([[0.0] * embedding_dim])
                else:
                    word_vector_tmp = np.array([getVector(sentence[word_index], embedding_dim)*dim_shrink])
                if word_index == 0:
                    word_vector = word_vector_tmp
                else:
                    word_vector = np.concatenate([word_vector, word_vector_tmp], 0)
            sentence_vector_tmp = np.array([word_vector])
            sentence_vector_dict[sentence_str] = sentence_vector_tmp
        sentence_vector_tmp = sentence_vector_dict.get(sentence_str)
        if maintain == 0:
            sentence_vector_dict.clear()
        if sentence_index == 0:
            sentence_vector = sentence_vector_tmp
        else:
            sentence_vector = np.concatenate([sentence_vector, sentence_vector_tmp], 0)
    embedded_chars = sentence_vector.astype(np.float32)
    return embedded_chars
