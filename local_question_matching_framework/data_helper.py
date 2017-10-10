import numpy as np
import sys
sys.path.append("..")
import random

# file path
# dataset_path = '/data/PycharmProjects/question_matching_framework/work_space/example/dataset/aaa'

def load_cn_data_from_files(classify_files):
    count = len(classify_files)
    x_text = []
    y = []
    for index in range(count):
        classify_file = classify_files[index]
        lines = list(open(classify_file, "r").readlines())
        label = [0] * count
        label[index] = 1
        labels = [label for _ in lines]
        if index == 0:
            x_text = lines
            y = labels
        else:
            x_text = x_text + lines
            y = np.concatenate([y, labels])
    x_text = [clean_str_cn(sent) for sent in x_text]
    return [x_text, y]

def clean_str_cn(string):
    """
    Tokenization/string cleaning for all datasets except for SST.
    Original taken from https://github.com/yoonkim/CNN_sentence/blob/master/process_data.py
    """
    return string.strip().lower()

def load_data(classify_files, config, sort_by_len=True, enhance = True, reverse=True):
    x_text, y = load_cn_data_from_files(classify_files)

    new_text = []
    if reverse == True:
        for text in x_text:
            text_list = text.strip().split(' ')
            text_list.reverse()
            reversed_text = ' '.join(text_list)
            new_text.append(reversed_text)
        x_text = new_text
    else:
        pass

    y = list(y)

    original_dataset = list(zip(x_text, y))

    if enhance == True:
        num_sample = len(original_dataset)

        # shuffle
        for i in range(num_sample):
            text_list = original_dataset[i][0].split(' ')
            random.shuffle(text_list)
            text_shuffled = ' '.join(text_list)
            label_shuffled = original_dataset[i][1]
            x_text.append(text_shuffled)
            y.append(label_shuffled)

    else:
        pass

    # Randomly shuffle data
    shuffle_indices = list(range(len(y)))
    random.shuffle(shuffle_indices)
    # print(shuffle_indices)
    x_shuffled = []
    y_shuffled_tmp = []
    for shuffle_indice in shuffle_indices:
        x_shuffled.append(x_text[shuffle_indice])
        y_shuffled_tmp.append(y[shuffle_indice])
    y_shuffled = np.array(y_shuffled_tmp)

    # train_set length
    n_samples = len(x_shuffled)
    # shuffle and generate train and valid data set
    sidx = np.random.permutation(n_samples)
    n_train = int(np.round(n_samples * (1. - config.valid_portion)))
    print("Train/Test split: {:d}/{:d}".format(n_train, (n_samples - n_train)))
    valid_set_x = [x_shuffled[s] for s in sidx[n_train:]]
    valid_set_y = [y_shuffled[s] for s in sidx[n_train:]]
    train_set_x = [x_shuffled[s] for s in sidx[:n_train]]
    train_set_y = [y_shuffled[s] for s in sidx[:n_train]]

    train_set = (train_set_x, train_set_y)
    valid_set = (valid_set_x, valid_set_y)
    # test_set = (x_test, y_test)

    # test_set_x, test_set_y = test_set
    valid_set_x, valid_set_y = valid_set
    train_set_x, train_set_y = train_set

    def len_argsort(seq):
        return sorted(range(len(seq)), key=lambda x: len(seq[x]))

    if sort_by_len:

        sorted_index = len_argsort(valid_set_x)
        valid_set_x = [valid_set_x[i] for i in sorted_index]
        valid_set_y = [valid_set_y[i] for i in sorted_index]

        sorted_index = len_argsort(train_set_x)
        train_set_x = [train_set_x[i] for i in sorted_index]
        train_set_y = [train_set_y[i] for i in sorted_index]

    train_set=(train_set_x,train_set_y)
    valid_set=(valid_set_x,valid_set_y)

    max_len = config.num_step

    def generate_mask(data_set):
        set_x = data_set[0]
        mask_x = np.zeros([max_len, len(set_x)])
        for i,x in enumerate(set_x):
            x_list = x.split(' ')
            if len(x_list) < max_len:
                mask_x[0:len(x_list), i] = 1
            else:
                mask_x[:, i] = 1
        new_set = (set_x, data_set[1], mask_x)
        return new_set

    train_set = generate_mask(train_set)
    valid_set = generate_mask(valid_set)

    train_data = (train_set[0], train_set[1], train_set[2])
    valid_data = (valid_set[0], valid_set[1], valid_set[2])

    return train_data, valid_data

# return batch data set
def batch_iter(data,batch_size, shuffle = True):
    # get data set and label
    x, y, mask_x = data

    # mask_x = np.array(mask_x)
    mask_x = np.asarray(mask_x).T.tolist()

    data_size = len(x)
    if shuffle:
        shuffle_indices = list(range(data_size))
        random.shuffle(shuffle_indices)
        shuffled_x = []
        shuffled_y = []
        shuffled_mask_x = []

        for shuffle_indice in shuffle_indices:
            shuffled_x.append(x[shuffle_indice])
            shuffled_y.append(y[shuffle_indice])
            shuffled_mask_x.append(mask_x[shuffle_indice])
    else:
        shuffled_x = x
        shuffled_y = y
        shuffled_mask_x = mask_x

    shuffled_mask_x = np.asarray(shuffled_mask_x).T  # .tolist()

    shuffled_x = np.array(shuffled_x)
    shuffled_y = np.array(shuffled_y)
    shuffled_mask_x = np.array(shuffled_mask_x)

    # num_batches_per_epoch=int((data_size-1)/batch_size) + 1
    num_batches_per_epoch = data_size // batch_size

    for batch_index in range(num_batches_per_epoch):
        start_index=batch_index*batch_size
        end_index=min((batch_index+1)*batch_size,data_size)
        return_x = shuffled_x[start_index:end_index]
        return_y = shuffled_y[start_index:end_index]
        return_mask_x = shuffled_mask_x[:,start_index:end_index]

        yield (return_x,return_y,return_mask_x)
