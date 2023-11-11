import numpy as np
import matplotlib.pyplot as plt
from statistics import mode

def equivalent_space(img, rho_resolution=1, theta_resolution=1):

    height, width = img.shape
    img_diagonal = np.ceil(np.sqrt(height**2 + width**2))
    rhos = np.arange(-img_diagonal, img_diagonal + 1, rho_resolution)
    thetas = np.deg2rad(np.arange(-90, 90, theta_resolution))

    H = np.zeros((len(rhos), len(thetas)), dtype=np.uint64)
    y_idxs, x_idxs = np.nonzero(img)

    for i in range(len(x_idxs)):
        x = x_idxs[i]
        y = y_idxs[i]

        for j in range(len(thetas)): # cycle through thetas and calc rho
            rho = int((x * np.cos(thetas[j]) +
                       y * np.sin(thetas[j])) + img_diagonal)
            H[rho, j] += 1

    return H, rhos, thetas

def voting(H, num_peaks, threshold=0, nhood_size=3):

    indices = []
    H1 = np.copy(H)
    for i in range(num_peaks):
        idx = np.argmax(H1) # find argmax in flattened array
        H1_idx = np.unravel_index(idx, H1.shape) # remap to shape of H
        indices.append(H1_idx)

        idx_y, idx_x = H1_idx
        if (idx_x - (nhood_size/2)) < 0: min_x = 0
        else: min_x = idx_x - (nhood_size/2)
        if ((idx_x + (nhood_size/2) + 1) > H.shape[1]): max_x = H.shape[1]
        else: max_x = idx_x + (nhood_size/2) + 1


        if (idx_y - (nhood_size/2)) < 0: min_y = 0
        else: min_y = idx_y - (nhood_size/2)
        if ((idx_y + (nhood_size/2) + 1) > H.shape[0]): max_y = H.shape[0]
        else: max_y = idx_y + (nhood_size/2) + 1

        min_x, max_x = int(np.ceil(min_x)), int(np.ceil(max_x))
        min_y, max_y = int(np.ceil(min_y)), int(np.ceil(max_y))

        for x in range(min_x, max_x):
            for y in range(min_y, max_y):
                H1[y, x] = 0

                if (x == min_x or x == (max_x - 1)):
                    H[y, x] = 255
                if (y == min_y or y == (max_y - 1)):
                    H[y, x] = 255

    return indices, H

def plot_equivalent_space(H, plot_title='Hough Accumulator Plot'):

    fig = plt.figure(figsize=(10, 10))
    fig.canvas.set_window_title(plot_title)

    plt.imshow(H, cmap='jet')

    plt.xlabel('Theta Direction'), plt.ylabel('Rho Direction')
    plt.tight_layout()
    plt.show()

def detected_lines(img, indicies, rhos, thetas):

    for i in range(len(indicies)):

        rho = rhos[indicies[i][0]]
        theta = thetas[indicies[i][1]]
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a*rho
        y0 = b*rho

        x1 = int(x0 + 1000*(-b))
        y1 = int(y0 + 1000*(a))
        x2 = int(x0 - 1000*(-b))
        y2 = int(y0 - 1000*(a))

        x_values = [x1, x2]
        y_values = [y1, y2]

        plt.close('all')
        plt.plot(x_values, y_values)
        plt.show()

def skewed_angle(img):

    A, rhos, thetas = equivalent_space(img)
    indices, Equiv = voting(A, 6, nhood_size=11)

    theta = []
    for i in range(len(indices)):
        theta.append(np.rad2deg(thetas[indices[i][1]]))

    angle = mode(theta)
    return angle

