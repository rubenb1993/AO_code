import numpy as np

def vec2str(vec):
    i = len(vec)
    if i != 1:
        return str(vec[0]) + '_' + vec2str(vec[1:])
    else:
        return str(vec[0])

a = np.array([1,2])

print(vec2str(a).type)
