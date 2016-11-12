import os
import random

data_dir = './a9a-data'
train_file = 'a9a'
test_file = 'a9a.t'
num_part = 4

def get_data(filename, is_shuffle=True):
    samples = []
    with open(filename, 'r') as f:
        for line in f:
            samples.append(line)
    if is_shuffle:
        random.shuffle(samples)
    return samples

train_dir = os.path.join(data_dir, 'train')
test_dir = os.path.join(data_dir, 'test')
model_dir = os.path.join(data_dir, 'models')

if not os.path.isdir(train_dir):
    os.mkdir(train_dir)
if not os.path.isdir(test_dir):
    os.mkdir(test_dir)
if not os.path.isdir(model_dir):
    os.mkdir(model_dir)

print('generating train data...')
samples = get_data(os.path.join(data_dir, train_file))
num_train = len(samples)
index = 1
part_size = int(num_train / num_part)
for part in range(num_part):
    with open(os.path.join(train_dir, 'part-00{}'.format(part + 1)), 'w') as f:
        for j in range(0, part_size):
            f.write(samples[index])
            index += 1

print('generating test data...')
samples = get_data(os.path.join(data_dir, test_file))
num_test = len(samples)
with open(os.path.join(test_dir, 'part-001'), 'w') as f:
    for i in range(0, num_test):
        f.write(samples[i])

print('done.')
