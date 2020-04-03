import numpy as np


def conv_2d(x, k):
    """
    Perform 2-D convolution
    :param x: input image
    :param k: kernel
    :return: the result of 2-D convolution
    """
    H, W = x.shape
    KH, KW = k.shape

    H_out = H - KH + 1
    W_out = W - KW + 1

    # Convert x into a matrix using indices
    i0 = np.repeat(np.arange(KH), KW)
    i1 = np.repeat(np.arange(H_out), W_out)
    i = i0.reshape(-1, 1) + i1.reshape(1, -1)
    print(i)

    j0 = np.tile(np.arange(KW), KH)
    j1 = np.tile(np.arange(W_out), H_out)
    j = j0.reshape(-1, 1) + j1.reshape(1, -1)
    print(j)

    x_crop = x[i, j]

    ## flatten kernel 
    k_crop = k.flatten()

    output = k_crop.dot(x_crop).reshape([H_out, W_out])

    return output


x1 = np.arange(1, 26, 1).reshape([5, 5])

k1 = np.arange(1, 5, 1).reshape([2, 2])


# print(conv_2d(x1,k1))


def conv_3d(x, k):
    """
    Perform 3-D convolution
    :param x: input image with size ([B,C1,H,W]). Otherwise, use np.transpose to reconstruct the shape as formulated.
    :param k: kernel with size ([C2,C1,KH,KW]). Otherwise, use np.transpose to reconstruct the shape as formulated.
    :return: the result of 3-D convolution in convolutional neural network.
    """
    B, C1, H, W = x.shape
    C2, _, KH, KW = k.shape

    H_out = H - KH + 1
    W_out = W - KW + 1

    i0 = np.repeat(np.arange(KH), KW)
    i0 = np.tile(i0, C1)
    i1 = np.repeat(np.arange(H_out), W_out)
    i = i0.reshape(-1, 1) + i1.reshape(1, -1)

    j0 = np.tile(np.arange(KW), KH * C1)
    j1 = np.tile(np.arange(W_out), H_out)
    j = j0.reshape(-1, 1) + j1.reshape(1, -1)

    m = np.repeat(np.arange(C1), KH * KW).reshape(-1, 1)

    x_crop = x[:, m, i, j]
    k_crop = k.reshape(C2, -1)

    output = k_crop.dot(x_crop).transpose(1, 0, 2)
    return output.reshape([B, C2, H_out, W_out])


x2 = np.arange(1, 49, 1).reshape([1, 3, 4, 4])
k2 = np.arange(1, 25, 1).reshape([2, 3, 2, 2])

res = conv_3d(x2, k2)


# print(res.shape)


def back_propagation_baseline(x, k, grad):
    x_grad = np.zeros_like(x)
    k_grad = np.zeros_like(k)
    B, H, W, C1 = x.shape
    KH, KW, _, C2 = k.shape
    Ho = H - KH + 1
    Wo = W - KW + 1
    ygrad = np.reshape(grad, [-1, C2])

    for i in range(KH):
        for j in range(KW):
            xij = np.matmul(ygrad, k[i, j, :, :].T)
            xij = np.reshape(xij, [B, Ho, Wo, C1])
            x_grad[:, i:(Ho + i), j:(Wo + j), :] = x_grad[:, i:(Ho + i),
                                                   j:(Wo + j), :] + xij
            kij = x[:, i:(Ho + i), j:(Wo + j), :]
            kij = np.reshape(kij, [-1, C1])
            k_grad[i, j, :, :] = k_grad[i, j, :, :] + np.matmul(kij.T, ygrad)

    return x_grad, k_grad


def back_propagation(x, k, grad):
    pass


x3 = x2.transpose(0, 2, 3, 1)
k3 = k2.transpose(2, 3, 1, 0)
grad = np.ones_like(res).transpose(0, 2, 3, 1)

# print(x3.shape, k3.shape, grad.shape)
x_grad_baseline, k_grad_baseline = back_propagation_baseline(x3, k3, grad)
print(x_grad_baseline.shape)
print('*' * 20)
print(k_grad_baseline.shape)
