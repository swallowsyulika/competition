import os
import random
import matplotlib.pyplot as plt
import config

dataset_dir = config.recognition_dataset_cache

data = {}

# check the number of samples of each classes
characters = os.listdir(dataset_dir)
random.shuffle(characters)

for character in characters[:100]:
    data[character] = len(os.listdir(os.path.join(dataset_dir, character)))

fig = plt.bar(range(len(data.values())), data.values(), tick_label=data.keys())
plt.tight_layout()
plt.show()


