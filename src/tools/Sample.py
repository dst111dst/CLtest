import numpy as np
import pandas as pd

def PersonalizedNeg(matrix, entity, start, end, neg_num):
    """
    matrix and entity should correspond with each other.
    :param matrix: <user, item>, for example,
    :param entity: current user id, for example,
    :param start: where randint begins,
    :param end: where randint ends
    """
    negatives = []
    for _ in range(neg_num):
        j = np.random.randint(start, end)
        while (entity, j) in matrix or j in negatives:
            j = np.random.randint(start, end)
        negatives.append(j)
    # for _ in range(hard_neg):
    return negatives


def MaskAttributes(items:int) -> list:
    negatives = list()
    return negatives

def Substitute(items:int,datas:pd.DataFrame) -> list:
    negatives = list()
    return negatives

def Positive(items, dataspos_num):
    pos = []


