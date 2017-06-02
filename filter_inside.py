import numpy as np
def filter_positions(inside, *args):
    """given a certain range of indices which are labeled inside, return an arbitrary number of vectors filtered with those indices"""
    new_positions = []
    for arg in args:
        new_positions.append(np.array(arg)[inside])
    return new_positions

def filter_nans(*args):
    """given vectors, filters out the NANs. Assumes that nans appear in the same place"""
    new_positions = []
    for arg in args:
        new_positions.append(arg[~np.isnan(arg)])
    return new_positions
