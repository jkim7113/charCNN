import keras
import model_util
import preprocess
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import random

# Model params
np.random.seed(123)

# Maximum length. Longer gets chopped. Shorter gets padded.
maxlen = 21
# Filters for conv layers
nb_filter = 32
# Number of units in the dense layer
dense_outputs = 256
# Conv layer kernel size
filter_kernels = [3, 3, 2, 2, 2, 2]
# Number of units in the final output layer. Number of classes.
cat_output = 1

vocab, _, vocab_size, _ = preprocess.create_vocab_set()

# Saved weight dir
model_weights_path = './crepe_model_weights_with_test_v1.0.weights.h5'

print('Loading model...')
model = model_util.create_model(filter_kernels, dense_outputs, maxlen, vocab_size,
                              nb_filter, cat_output)

model.load_weights(model_weights_path)

# samples = ["fast", "speedy", "swift", "rapid", "expeditious", "alacritous", "simultaneously"]

# prediction = model.predict(preprocess.encode_data(samples, maxlen, vocab))
# for key, word in enumerate(samples):
#     print(word + ": " + str(prediction[key].item()))

def test():
    data = pd.read_csv('wordlist.csv')
    data = data.dropna()
    data["Word"] = data["Word"].str.lower().replace("'", "")

    words = data["Word"].tolist()
    pred = model.predict(preprocess.encode_data(words, maxlen, vocab))

    sns.histplot(pred, kde=True, bins=30, color="blue")

    plt.title('Distribution of Word Difficulty')
    plt.xlabel('Word Difficulty')
    plt.ylabel('Frequency') 

    plt.show()

# test()

data = pd.read_csv('wordlist.csv')
data = data.dropna()
data["Word"] = data["Word"].str.lower().replace("'", "")
words = data["Word"].tolist()
inputs = [""]

for i in range(10):
    num = random.randint(0, len(words))
    inputs.append(words[num])

prediction = model.predict(preprocess.encode_data(inputs, maxlen, vocab))
for key, word in enumerate(inputs):
    print(word + ": " + str(prediction[key].item()))


inputs = input("Enter words: ").split()

prediction = model.predict(preprocess.encode_data(inputs, maxlen, vocab))
for key, word in enumerate(inputs):
    print(word + ": " + str(prediction[key].item()))