import matplotlib.pyplot as plt
import numpy as np
from data.datasets import MNISTDataset


def main():
    N = 5
    dataset = MNISTDataset('./data/download/')
    np.random.seed(69)
    indices = np.random.randint(0, len(dataset), N)

    images = [dataset[idx][0]['x'] for idx in indices]

    fig, axs = plt.subplots(1, N, figsize=(15, 3))
    for idx, img in enumerate(images):
        axs[idx].imshow(np.array(img), cmap=plt.cm.binary)
        axs[idx].set_xticks([])
        axs[idx].set_yticks([])
    plt.tight_layout()
    plt.savefig('mnist.png')
    plt.show()


if __name__ == "__main__":
    main()
