import numpy as np

def conv_nested(image, kernel):
    """A naive implementation of convolution filter.

    This is a naive implementation of convolution using 4 nested for-loops.
    This function computes convolution of an image with a kernel and outputs
    the result that has the same shape as the input image.

    Args:
        image: numpy array of shape (Hi, Wi).
        kernel: numpy array of shape (Hk, Wk).

    Returns:
        out: numpy array of shape (Hi, Wi).
    """
    Hi, Wi = image.shape
    Hk, Wk = kernel.shape
    out = np.zeros((Hi, Wi))

    ### YOUR CODE HERE

    for i in range(0, Hi):
        for j in range(0, Wi):
            for m in range(0, Hk):
                for n in range(0, Wk):
                    if Hi > i - m + Hk//2 >= 0 and Wi > j - n + Wk//2 >= 0:
                        out[i, j] += image[i - (m - Hk//2), j - (n - Wk//2)] * kernel[m, n]

    ### END YOUR CODE

    return out

def zero_pad(image, pad_height, pad_width):
    """ Zero-pad an image.

    Ex: a 1x1 image [[1]] with pad_height = 1, pad_width = 2 becomes:

        [[0, 0, 0, 0, 0],
         [0, 0, 1, 0, 0],
         [0, 0, 0, 0, 0]]         of shape (3, 5)

    Args:
        image: numpy array of shape (H, W).
        pad_width: width of the zero padding (left and right padding).
        pad_height: height of the zero padding (bottom and top padding).

    Returns:
        out: numpy array of shape (H+2*pad_height, W+2*pad_width).
    """

    H, W = image.shape
    out = np.zeros((H + 2 * pad_height, W + 2 * pad_width))

    ### YOUR CODE HERE

    out[pad_height: -pad_height, pad_width: -pad_width] = image

    ### END YOUR CODE

    return out


def conv_fast(image, kernel):
    """ An efficient implementation of convolution filter.

    This function uses element-wise multiplication and np.sum()
    to efficiently compute weighted sum of neighborhood at each
    pixel.

    Hints:
        - Use the zero_pad function you implemented above
        - There should be two nested for-loops
        - You may find np.flip() and np.sum() useful

    Args:
        image: numpy array of shape (Hi, Wi).
        kernel: numpy array of shape (Hk, Wk).

    Returns:
        out: numpy array of shape (Hi, Wi).
    """
    Hi, Wi = image.shape
    Hk, Wk = kernel.shape
    out = np.zeros((Hi, Wi))

    ### YOUR CODE HERE

    padded_img = zero_pad(image, Hk//2, Wk//2)
    kernel = np.flip(kernel)
    for i in range(Hk//2, Hi + Hk//2):
        for j in range(Wk//2, Wi + Wk//2):
            i_begin, j_begin = i - (Hk//2), j - (Wk//2)
            out[i - Hk//2, j - Wk//2] = np.sum(padded_img[i_begin: i_begin + Hk, j_begin: j_begin + Wk] * kernel)

    ### END YOUR CODE

    return out

def conv_faster(image, kernel):
    """
    Args:
        image: numpy array of shape (Hi, Wi).
        kernel: numpy array of shape (Hk, Wk).

    Returns:
        out: numpy array of shape (Hi, Wi).
    """
    Hi, Wi = image.shape
    Hk, Wk = kernel.shape
    out = np.zeros((Hi, Wi))

    ### YOUR CODE HERE
    
    # Solution uses Convolution Theorem
    padded_image = zero_pad(image, Hk//2, Wk//2)
    Fi = np.fft.rfft2(padded_image)
    Fk = np.fft.rfft2(kernel, padded_image.shape)
#     out = np.around(np.real(np.fft.ifft2(np.fft.fft2(padded_image) * np.fft.fft2(kernel, padded_image.shape))), 10)
    out = np.fft.irfft2(Fi * Fk)[(Hk + 1)//2:, (Wk + 1)//2:]

    ### END YOUR CODE

    return out

def cross_correlation(f, g):
    """ Cross-correlation of f and g.

    Hint: use the conv_fast function defined above.

    Args:
        f: numpy array of shape (Hf, Wf).
        g: numpy array of shape (Hg, Wg).

    Returns:
        out: numpy array of shape (Hf, Wf).
    """

    out = np.zeros_like(f)
    
    ### YOUR CODE HERE
    
    out = conv_fast(f, np.flip(g))
    
    ### END YOUR CODE

    return out


#NOTE: Since cross_correlation is not normalized to 1, optimal threshold value was found to be ~1100 
def zero_mean_cross_correlation(f, g):
    """ Zero-mean cross-correlation of f and g.

    Subtract the mean of g from g so that its mean becomes zero.

    Hint: you should look up useful numpy functions online for calculating the mean.

    Args:
        f: numpy array of shape (Hf, Wf).
        g: numpy array of shape (Hg, Wg).

    Returns:
        out: numpy array of shape (Hf, Wf).
    """

    out = np.zeros_like(f)
    
    ### YOUR CODE HERE
    
    g = g - np.mean(g)
    out = cross_correlation(f, g)
    
    ### END YOUR CODE

    return out

def normalized_cross_correlation(f, g):
    """ Normalized cross-correlation of f and g.

    Normalize the subimage of f and the template g at each step
    before computing the weighted sum of the two.

    Hint: you should look up useful numpy functions online for calculating 
          the mean and standard deviation.

    Args:
        f: numpy array of shape (Hf, Wf).
        g: numpy array of shape (Hg, Wg).

    Returns:
        out: numpy array of shape (Hf, Wf).
    """

    ### YOUR CODE HERE

    Hi, Wi = f.shape
    Hk, Wk = g.shape
    out = np.zeros((Hi, Wi))


    padded_f = zero_pad(f, Hk//2, Wk//2)
    g_normalized = (g - np.mean(g))/np.std(g)
    for i in range(Hk//2, Hi + Hk//2):
        for j in range(Wk//2, Wi + Wk//2):
            i_begin, j_begin = i - (Hk//2), j - (Wk//2)
            padded_slice = padded_f[i_begin: i_begin + Hk, j_begin: j_begin + Wk]
            padded_normalized = (padded_slice - np.mean(padded_slice))/np.std(padded_slice)
            out[i - Hk//2, j - Wk//2] = np.sum(padded_normalized * g_normalized)
            
    ### END YOUR CODE

    return out
