
import numpy as np

import skimage
import cv2


def abstractify(img: np.array, sigma_color=5.5, sigma_space=3.9,
                Nq=15, 
                tau=0.99, phi=0.75) -> np.array:
    # change to Lab color space
    lab = skimage.color.rgb2lab(img)

    # Bilateral filtering
    bf = lab
    for i in range(7):
        bf = cv2.bilateralFilter(bf, d=5, sigmaColor=sigma_color, sigmaSpace=sigma_space, 
                                 borderType=cv2.BORDER_REFLECT_101)

    # Quantization
    bf_q = bf.copy()
    Dq = 100/Nq

    L = bf_q[..., 0]
    q = (L // Dq) * Dq
    L = q + (np.tanh((L-q)/Dq*10 - 5) + 1) * Dq/2

    bf_q[..., 0] = L

    # Edge enhancement
    L = bf[..., 0].copy()

    L_sigma1 = cv2.GaussianBlur(L, ksize=[0,0], sigmaX=2.0, borderType=cv2.BORDER_REFLECT_101)
    L_sigma2 = cv2.GaussianBlur(L, ksize=[0,0], sigmaX=2.53, borderType=cv2.BORDER_REFLECT_101)

    diff = L_sigma1-tau*L_sigma2
    E = np.where(diff > 0, np.ones_like(L), 1 + np.tanh(phi*diff))

    bf_q[..., 0] *= E

    return skimage.color.lab2rgb(bf_q)