from __future__ import division  # floating point division
import numpy as np

def threshold_probs(probs):
    """ Converts probabilities to hard classification """
    classes = np.ones(len(probs),)
    classes[probs < 0.5] = 0
    return classes

def update_dictionary_items(dict1, dict2):
    """ Replace any common dictionary items in dict1 with the values in dict2 
    There are more complicated and efficient ways to perform this task,
    but we will always have small dictionaries, so for our use case, this simple
    implementation is acceptable.
    """
    for k in dict1:
        if k in dict2:
            dict1[k]=dict2[k]