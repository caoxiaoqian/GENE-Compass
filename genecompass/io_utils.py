import pandas as pd
from typing import List, Dict
from joblib import Parallel, delayed
from tqdm import tqdm
import numpy as np
from Bio import SeqIO
    


############################some_tools for rose gene##############
def ncbi2mrna(ncbi_id):
    ncbi_id = ncbi_id.replace('m_C', 'mC')
    return f'mRNA:{ncbi_id}'

def mrna2ncbi(mrna_id):
    if mrna_id.startswith('mRNA:'):
        ncbi_id = mrna_id[5:]  # Remove the 'mRNA:' prefix
        ncbi_id = ncbi_id.replace('mC', 'm_C')  # Reverse the replacement done in ncbi2mrna
        return ncbi_id
    return mrna_id  # Return as is if it doesn't have the expected prefix


########################## grn tools #################################
def for_datafream(net_adj_matrix, tf):
    net_adj_matrix = net_adj_matrix.loc[tf, :]
    net_adj_matrix = pd.DataFrame(net_adj_matrix)
    net_adj_matrix['Target'] = net_adj_matrix.index
    net_adj_matrix['TF'] = [tf] * len(net_adj_matrix)
    net_adj_matrix['abs_importance'] = abs(net_adj_matrix[tf])
    net_adj_matrix = net_adj_matrix.sort_values(by='abs_importance', ascending=False)
    net_adj_matrix['rank'] = [i for i in range(len(net_adj_matrix))]
    return net_adj_matrix