def conv2dtrans_out_shape(H_in, k, s, p_i, p_o, d):
    """
    Compute the output channel shape of a list of sequential conv2dtranspose layers. 
    For all parameters not the input height and width, (i.e., k, s, p_i, p_o, d), if 2D array is not 

    Parameters:
        H_in (List[int] or int): input height and width, accepts either a list of size (2) to specify rectangle width and height
        k (List[int]): kernel size.
        s (List[int]): stride.
        p_i (List[int]): input padding.
        p_o (List[int]): output padding.
        d (List[int]): dilation.

    Returns:
        list : output height and width.
    """
    if len(H_in) == 1:
        H_in = (H_in[0], H_in[0])
    if len(k) == 1:
        k = (k[0], k[0])
    if len(s) == 1:
        s = (s[0], s[0])
    if len(p_i) == 1:
        p_i = (p_i[0], p_i[0])
    if len(p_o) == 1:
        p_o = (p_o[0], p_o[0])
    if len(d) == 1:
        d = (d[0], d[0])

    H_out = H_in[0]
    W_out = H_in[1]

    for i in range(len(k)):
        H_out = int( (H_out - 1)*s[i][0] - 2*p_i[i][0] + d[i][0]*(k[i][0] - 1) + p_o[i][0] + 1)
        W_out = int( (W_out - 1)*s[i][1] - 2*p_i[i][1] + d[i][1]*(k[i][1] - 1) + p_o[i][1] + 1)

    return (H_out, W_out)

def conv2dtrans_in_shape(H_out, k, s, p_i, p_o, d):
    """
    Compute the input shape of a conv2dtranspose layer.

    Parameters:
        H_out (List[int] or int): output height and width, accepts either a list of size (2) to specify rectangle width and height
        k (List[int]): kernel size.
        s (List[int]): stride.
        p_i (List[int]): input padding.
        p_o (List[int]): output padding.
        d (List[int]): dilation.

    Returns:
        List[int] : input height and width.
    """
    if len(H_out) == 1:
        H_out = (H_out[0], H_out[0])
    if len(k) == 1:
        k = (k[0], k[0])
    if len(s) == 1:
        s = (s[0], s[0])
    if len(p_i) == 1:
        p_i = (p_i[0], p_i[0])
    if len(p_o) == 1:
        p_o = (p_o[0], p_o[0])
    if len(d) == 1:
        d = (d[0], d[0])
    H_in = H_out[0]
    W_in = H_out[1]

    for i in reversed(range(len(k))):
        H_in = int( (H_in + 2*p_i[i][0] - d[i][0]*(k[i][0] - 1) - p_o[i][0] - 1) / s[i][0]) + 1
        W_in = int( (W_in + 2*p_i[i][1] - d[i][1]*(k[i][1] - 1) - p_o[i][1] - 1) / s[i][1]) + 1

    return (H_in, W_in)

def conv2d_out_shape(H_in, k, s, p_i, d):
    """
    Compute the output channel shape of a list of sequential conv2d layers. 
    For all parameters not the input height and width, (i.e., k, s, p_i, p_o, d), if 2D array is not 

    Parameters:
        H_in (List[int] or int): input height and width, accepts either a list of size (2) to specify rectangle width and height
        k (List[int]): kernel size.
        s (List[int]): stride.
        p_i (List[int]): input padding.
        p_o (List[int]): output padding.
        d (List[int]): dilation.

    Returns:
        list : output height and width.
    """
    if len(H_in) == 1:
        H_in = (H_in[0], H_in[0])
    if len(k) == 1:
        k = (k[0], k[0])
    if len(s) == 1:
        s = (s[0], s[0])
    if len(p_i) == 1:
        p_i = (p_i[0], p_i[0])
    if len(p_o) == 1:
        p_o = (p_o[0], p_o[0])
    if len(d) == 1:
        d = (d[0], d[0])

    H_out = H_in[0]
    W_out = H_in[1]

    for i in range(len(k)):

        H_out = int((H_out + 2*p_i[i][0] - d[i][0]*(k[i][0] - 1) - 1) / s[i][0]) + 1
        W_out = int((W_out + 2*p_i[i][1] - d[i][1]*(k[i][1] - 1) - 1) / s[i][1]) + 1
    
    return (H_out, W_out)

def conv2d_in_shape(H_out, k, s, p_i, d):
    """
    Compute the input shape of a conv2d layer.

    Parameters:
        H_out (List[int] or int): output height and width, accepts either a list of size (2) to specify rectangle width and height
        k (List[int]): kernel size.
        s (List[int]): stride.
        p_i (List[int]): input padding.
        p_o (List[int]): output padding.
        d (List[int]): dilation.

    Returns:
        List[int] : input height and width.
    """
    if len(H_out) == 1:
        H_out = (H_out[0], H_out[0])
    if len(k) == 1:
        k = (k[0], k[0])
    if len(s) == 1:
        s = (s[0], s[0])
    if len(p_i) == 1:
        p_i = (p_i[0], p_i[0])
    if len(p_o) == 1:
        p_o = (p_o[0], p_o[0])
    if len(d) == 1:
        d = (d[0], d[0])
    H_in = H_out[0]
    W_in = H_out[1]
    

    for i in reversed(range(len(k))):
        H_in = (H_in - 1)*s[i][0] + 1 - 2*p_i[i][0] + d[i][0]*(k[i][0] - 1) 
        W_in = (W_in - 1)*s[i][1] + 1 - 2*p_i[i][1] + d[i][1]*(k[i][1] - 1) 
    
    return (H_in, W_in)

