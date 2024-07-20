import string
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from keras.utils import to_categorical
import seaborn as sns
from scipy import stats
import random

def standardize(data, mean=0.5, std=(0.5/3)):
    shifted_data = data - data.min() + 1
    box_cox_transformed, _ = stats.boxcox(shifted_data)
    box_cox_transformed = mean + (box_cox_transformed - box_cox_transformed.mean())*(std / box_cox_transformed.std())
    return box_cox_transformed

def normalize(data, mean=0.5, std=(0.5/3)):
    return mean + (data - data.mean())*(std / data.std())

def visualize():
    data = pd.read_csv('I159729.csv')
    data = data.dropna()
    i_zscore = (data['I_Zscore'])
    data["Normalized_HAL_Freq"] = stats.zscore(data['Log_Freq_HAL'])
    inversed_freq = -data["Normalized_HAL_Freq"]

    difficulty = normalize((i_zscore + 2 * inversed_freq) / 2)

    _, axis = plt.subplots(ncols=3)
    sns.histplot(i_zscore, kde=True, bins=30, color="blue", ax=axis[0])
    sns.histplot(inversed_freq, kde=True, bins=30, color="green", ax=axis[1])
    sns.histplot(difficulty, kde=True, bins=100, color="red", ax=axis[2])

    axis[0].axvline(x=i_zscore.mean(), color='red', linestyle='--', label="mean")
    axis[0].axvline(x=np.median(i_zscore), color='green', linestyle='--', label="median")
    axis[2].axvline(x=i_zscore.mean(), color='red', linestyle='--', label=i_zscore.mean())
    # Add titles and labels
    axis[0].set_title('Distribution of I-Zscore')
    axis[0].set_xlabel('I-Zscore')
    axis[0].set_ylabel('Frequency') 

    axis[1].set_title('Distribution of Log Freq HAL Score')
    axis[1].set_xlabel('Log_Freq_HAL')
    axis[1].set_ylabel('Frequency') 

    axis[2].set_title('Distribution of word difficulty (theta)')
    axis[2].set_xlabel('Difficulty')
    axis[2].set_ylabel('Frequency') 

    # Show the plot
    plt.show()

# visualize() # For testing purposes only

def load_data():
    data = pd.read_csv('I159729.csv')
    data = data.dropna()
    data.drop(data.columns[[1, 2, 4, 6, 7, 8]], axis=1, inplace=True)
    data["Word"] = data["Word"].str.lower().replace("'", "")
    
    # data["Difficulty"] = 0
    # data.loc[data["I_Zscore"] > 0, ["Difficulty"]] = 1

    i_zscore = (data['I_Zscore'])
    data["Normalized_HAL_Freq"] = stats.zscore(data['Log_Freq_HAL'])
    inversed_freq = -data["Normalized_HAL_Freq"]

    data["Difficulty"] = normalize((i_zscore + 2 * inversed_freq) / 2)
    
    data_x = data["Word"]
    data_x = np.array(data_x)
    data_y = data["Difficulty"]
    data_y = np.array(data_y)
    # print(data.sort_values(by='Difficulty', ascending=True)[['Word', 'Difficulty']].head(20))

    return data_x, data_y

# data_x, data_y = load_data()
# for i in range(10):
#     num = random.randint(0, len(data_x))
#     print(num, data_x[num], data_y[num])

# print(data_x[36135], data_y[36135])
# print(data_x[2790], data_y[2790])
# print(data_x[24137], data_y[24137])
# print(data_x[35675], data_y[35675])
# print(data_x[18277], data_y[18277])
# print(data_x[5291], data_y[5291])
# print(data_x[9286], data_y[9286])
# print(data_x[18622], data_y[18622])
# print(data_x[5606], data_y[5606])
# print(data_x[29447], data_y[29447])

def encode_data(x, maxlen, vocab):
    # Iterate over the loaded data and create a matrix of size (len(x), maxlen)
    # Each character is encoded into a one-hot array later at the lambda layer.
    # Chars not in the vocab are encoded as -1, into an all zero vector.

    input_data = np.zeros((len(x), maxlen), dtype=np.int64)
    for dix, sent in enumerate(x):
        counter = 0
        for c in sent:
            if counter >= maxlen:
                pass
            else:
                ix = vocab.get(c, -1)  # get index from vocab dictionary, if not in vocab, return -1
                input_data[dix, counter] = ix
                counter += 1
    return input_data


def create_vocab_set():

    alphabet = set(list(string.ascii_lowercase))
    vocab_size = len(alphabet)
    vocab = {}
    reverse_vocab = {}
    for ix, t in enumerate(alphabet):
        vocab[t] = ix
        reverse_vocab[ix] = t

    return vocab, reverse_vocab, vocab_size, alphabet
    
'''
data_x, data_y = load_data()
print(data_x)
print(data_y)
# Max len -> 21

vocab, reverse_vocab, vocab_size, alphabet = create_vocab_set()
print(vocab)

input_data = encode_data(data_x, 21, vocab)
print(input_data)
'''
