import tensorflow as tf


def read_csv(filename):
    with open(filename, 'r') as f:
        for line in f.readlines():
            record = line.rstrip().split(',')
            features = [float(n) for n in record[:-1]]
            label = int(record[-1])

            yield features, label

def get_dataset():
    filename = 'train_val_list.csv'
    generator = lambda: read_csv(filename)

    return tf.data.Dataset.from_generator(generator=generator, )
