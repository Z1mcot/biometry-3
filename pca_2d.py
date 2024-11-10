import cv2
import cupy as cp
import numpy as np
import matplotlib.pyplot as plt
import glob


def load_dataset():
    faces = []
    indices = []

    for f in glob.iglob('ORLdataset/**/**'):
        faces.append(cp.array(cv2.imread(f, cv2.IMREAD_GRAYSCALE)).flatten().tolist())
        index = int(f.split("\\")[-2].split("s")[1])
        indices.append(index)

    dataset = [(faces[i], indices[i]) for i in range(len(indices))]
    dataset.sort(key=lambda x: x[1])

    return dataset

class PlotData2D:
    def __init__(self, train_x, train_y, test_x, test_y, top_k_eigs, dataset_weights, test_weights):
        self.train_y = train_y
        self.train_x = train_x
        self.test_x = test_x
        self.test_y = test_y
        self.top_k_eigs = top_k_eigs
        self.dataset_weights = dataset_weights
        self.test_weights = test_weights

def detect():
    data_set = load_dataset()
    
    train = []
    test = []
    
    for i, item in enumerate(data_set):
        if i % 10 > 7:
            test.append(item)
        else:
            train.append(item)
    
    train_x = cp.array([X[0] + [X[1]] for X in train], dtype=float)[:, :-1].T
    train_x = train_x - cp.mean(train_x, axis=1).reshape((-1, 1))
    train_y = cp.array([X[0] + [X[1]] for X in train])[:, -1]
    
    test_x = cp.array([X[0] + [X[1]] for X in test], dtype=float)[:, :-1].T
    test_x = test_x - cp.mean(test_x, axis=1).reshape((-1, 1))
    test_y = cp.array([X[0] + [X[1]] for X in test])[:, -1]
    
    # Ковариационная матрица
    cov_mat = train_x.T @ train_x
    w, v = np.linalg.eig(cov_mat.get())
    
    # Вычисление количества главных компонент
    val_sum = cp.sum(w)
    running_sum = 0
    k = int()
    count = 0
    for i in range(w.shape[0]):
        running_sum += w[i]
        if running_sum >= .9 * val_sum:
            k = i
            break
        count += 1
    print(count)
    
    sorted_idx = cp.flip(cp.argsort(w))
    
    top_k_eigs = v[:, sorted_idx][:, :k]
    
    # Transform eigenvectors back to original space. Av = u
    
    top_k_eigs = train_x.get() @ top_k_eigs
    
    # Get weights for images in train set, each row is a training image's weights
    dataset_weights = cp.array(train_x.T.get() @ top_k_eigs)
    
    # Test it out
    test_weights = cp.array(test_x.T.get() @ top_k_eigs)

    return PlotData2D(train_x, train_y, test_x, test_y, top_k_eigs, dataset_weights, test_weights)

def plot(data):
    # "Среднее" лицо
    plt.imshow(cp.mean(data.train_x, axis=1).reshape((-1, 92)).get(), cmap='Greys')

    eigenfaces_fig, eigenfaces_ax = plt.subplots(4, 4)

    for i in range(16):
        eigenfaces_ax[int(i / 4)][i % 4].imshow(data.top_k_eigs[:, i].reshape(112, 92), cmap="gray")

    eigenfaces_fig.set_size_inches(10, 10)

    fig, ax = plt.subplots(5, 2)
    guessed_correctly = 0
    for i in range(data.test_weights.shape[0]):
        distances = cp.linalg.norm(data.dataset_weights - data.test_weights[i, :], axis=1)
        idx_max_match = cp.argmin(distances)
        if data.test_y[i] == data.train_y[idx_max_match]:
            guessed_correctly += 1
        if i < 5:
            ax[i][0].imshow(data.test_x[:, i].reshape(112, 92).get(), cmap="gray")
            ax[i][1].imshow(data.train_x[:, idx_max_match].reshape(112, 92).get(), cmap="gray")
            ax[i][0].set_title("Исходное изображение")
            ax[i][1].set_title("Ближайшее совпадение")

    print(f"Точность распознавания: {guessed_correctly / data.test_weights.shape[0]}")

    fig.set_size_inches(10, 30)