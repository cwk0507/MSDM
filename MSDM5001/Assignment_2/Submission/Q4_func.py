import numpy as np

var_dict={}

def init_dict(X, X_shape):
    var_dict['X'] = X
    var_dict['X_shape'] = X_shape

def update(grid_point):
    X_np = np.frombuffer(var_dict['X']).reshape(var_dict['X_shape'])
    last = var_dict['X_shape'][0]
    for k in range(1000):
        for item in grid_point:
            i = item[0]
            j = item[1]
            if j != 0 and j != last-1 and i != 0 and i != last-1:
                X_np[i,j] = (1/4)*(X_np[i-1,j]+X_np[i+1,j]+X_np[i,j-1]+X_np[i,j+1])