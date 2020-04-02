import numpy as np

## 2-D convolution
def conv_2d(x, k):
    H, W = x.shape
    KH, KW = k.shape

    H_out = H - KH + 1
    W_out = W - KW + 1

    # Convert x into a matrix using indices
    i0 = np.repeat(np.arange(KH), KW)
    i1 = np.repeat(np.arange(H_out), W_out)
    i = i0.reshape(-1,1) + i1.reshape(1,-1)
    print(i)

    j0 = np.tile(np.arange(KW), KH)
    j1 = np.tile(np.arange(W_out), H_out)
    j = j0.reshape(-1,1) + j1.reshape(1,-1)
    print(j)

    x_crop = x[i,j]

    ## flatten kernel 
    k_crop = k.flatten()

    output = k_crop.dot(x_crop).reshape([H_out, W_out])
    
    return output

x1 = np.arange(1,26,1).reshape([5,5])

k1 = np.arange(1,5,1).reshape([2,2])

# print(conv_2d(x1,k1))

## 3-D convolution using index 
def conv_3d(x, k):
    B, C1, H, W = x.shape
    C2, _, KH, KW = k.shape

    H_out = H - KH + 1
    W_out = W - KW + 1

    i0 = np.repeat(np.arange(KH), KW)
    i0 = np.tile(i0,C1)
    i1 = np.repeat(np.arange(H_out), W_out)
    i = i0.reshape(-1,1) + i1.reshape(1,-1)
    # print(i)

    j0 = np.tile(np.arange(KW), KH*C1)
    j1 = np.tile(np.arange(W_out), H_out)
    j = j0.reshape(-1,1) + j1.reshape(1,-1)
    # print(j)

    m = np.repeat(np.arange(C1), KH * KW).reshape(-1,1)
    # print(m)

    x_crop = x[:,m,i,j]
    k_crop = k.reshape(C2,-1)

    output = k_crop.dot(x_crop).transpose(1,0,2)
    return output.reshape([B, C2, H_out, W_out])


x2 = np.arange(1,97,1).reshape([2,3,4,4])
k2 = np.arange(1,25,1).reshape([2,3,2,2])

res = conv_3d(x2, k2)
print(res)
print(conv_3d(x2[1,:,:,:].reshape([1,x2.shape[1],x2.shape[2],x2.shape[3]]), k2))
# print(x2[0,:,:,:])