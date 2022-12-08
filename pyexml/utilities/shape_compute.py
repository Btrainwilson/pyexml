
def conv2dtrans_out_shape(H_in, k, s, p_i, p_o, d):
    H_out = H_in
    for i in range(len(k)):
        H_out = (H_in - 1)*s[i] - 2*p_i[i] + d[i]*(k[i] - 1) + p_o[i] + 1
    
    return H_out

def conv2dtrans_in_shape(H_out, k, s, p_i, p_o, d):
    H_in = H_out
    for i in reversed(range(len(k))):
        H_in = int( (H_in + 2*p_i[i] - d[i]*(k[i] - 1) - p_o[i] - 1) / s[i]) + 1
    
    return H_in


def conv2d_out_shape(H_in, k, s, p_i, d):
    H_out = H_in
    for i in range(len(k)):
        H_out = int((H_in + 2*p_i[i] - d[i]*(k[i] - 1) - 1) / s[i]) + 1
    
    return H_out

def conv2d_in_shape(H_out, k, s, p_i, d):
    H_in = H_out
    for i in reversed(range(len(k))):
        H_in = (H_in - 1)*s[i] + 1 - 2*p_i[i] + d[i]*(k[i] - 1) 
    
    return H_in

