import cupy as cp
import cv2
import os
import matplotlib.pyplot as plt

def load_orl_faces():
    face_images = []
    for i in range(1, 41):
        for j in range(1, 11):
            img_path = os.path.join('ORLdataset', f"s{i}", f"{j}.pgm")
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if img is not None:
                face_images.append(img.flatten())
    return cp.array(face_images)

# PCA для метода с уменьшением изображения
def pca_image_reduction(x):
    x_centered = x - cp.mean(x, axis=0)

    cov_matrix = cp.cov(x_centered.T)
    eigenvalues, _ = cp.linalg.eigh(cov_matrix)

    return cov_matrix, eigenvalues

# PCA с использованием матрицы Грамма-Шмидта
def pca_gram_schmidt(x):
    x_centered = x - cp.mean(x, axis=0)

    gram_matrix = cp.dot(x_centered, x_centered.T)
    q, _ = cp.linalg.qr(gram_matrix)

    eigenvalues, _ = cp.linalg.eigh(q)
    return gram_matrix, eigenvalues

class PlotData:
    def __init__(self, cov_matrix_image, eigenvalues_image, cov_matrix_gram, eigenvalues_gram):
        self.cov_matrix_image = cov_matrix_image
        self.eigenvalues_image = eigenvalues_image
        self.cov_matrix_gram = cov_matrix_gram
        self.eigenvalues_gram = eigenvalues_gram

def detect():
    images = load_orl_faces()

    # вычисление матриц ковариации и собственных чисел для двух методов
    cov_matrix_image, eigenvalues_image = pca_image_reduction(images)
    cov_matrix_gram, eigenvalues_gram = pca_gram_schmidt(images)

    return PlotData(
        cov_matrix_image.get(),
        eigenvalues_image.get(),
        cov_matrix_gram.get(),
        eigenvalues_gram.get()
    )

def plot(data):
    plt.show()
    # визуализация результатов
    plt.figure(figsize=(15, 10))

    # отображение первой матрицы ковариации
    plt.subplot(2, 2, 1)
    plt.imshow(data.cov_matrix_image, cmap='gray')
    plt.title('Ковариационная матрица (уменьшение изображения)')

    # отображение второй матрицы ковариации
    plt.subplot(2, 2, 2)
    plt.imshow(data.cov_matrix_gram, cmap='gray')
    plt.title('Ковариационная матрица (преобразование Грамма-Шмидта)')

    # отображение графика собственных чисел для первого метода
    plt.subplot(2, 2, 3)
    plt.plot(data.eigenvalues_image)
    plt.title('Собственные числа (уменьшение изображения)')

    # отображение графика собственных чисел для второго метода
    plt.subplot(2, 2, 4)
    plt.plot(data.eigenvalues_gram)
    plt.title('Собственные числа (преобразование Грамма-Шмидта)')

    plt.tight_layout()
    plt.show()

