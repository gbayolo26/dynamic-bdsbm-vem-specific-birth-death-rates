#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 19 18:02:02 2025

@author: gabriela
"""
import numpy as np
import pandas as pd
import random
from copy import copy

        
def data_generation(lamb, mu, t_end, L, K, pi, snapshots_times, seed = 1):
    
    """
    Generate data from the Birth–Death Stochastic Block Model (BD-SBM).

    This function first simulates a birth–death process to generate individual
    lifetimes and community memberships, then generates interaction graphs at
    specified snapshot times according to an SBM with probability matrix `pi`.

    Parameters
    ----------
    lamb : float
        Birth rate.
    mu : float
        Death rate.
    t_end : float
        Final time (end of the simulation horizon).
    L : array-like
        Initial number of individuals per community.
    K : int
        Number of communities (K = len(L[0])).
    pi : array-like
        SBM connection probability matrix (shape: K x K).
    snapshots_times : array-like
        Snapshot times at which graphs are generated.

    Returns
    -------
    interaction_counts :  array-like, shape (N, N)
        Pairwise interaction counts over the whole study interval. Entry (i, j) is the 
        total number of observed interactions between individuals i and j.
    alive_matrix : array-like, shape (N, N)
        Pairwise co-aliveness (exposure) matrix. Entry (i, j) is the number of
        observation times at which individuals i and j are simultaneously alive/active
        (i.e., the number of opportunities for them to interact).
    df : pandas.DataFrame
        Per-individual metadata (community assignment and birth/death information).
    tau : array-like
        Vector of ordered birth/death event times.
    b : array-like, shape (len(tau),)
        Vector of event types: 1 for a birth event, -1 for a death event.    
    N_l : array-like, shape (len(tau),)
        Number of alive individuals immediately after each birth/death event time in `tau`.
    """
    
    df = birth_death_function(lamb, mu, t_end, L, K, seed)
   
    interaction_counts, non_interaction_counts = discrete_graph_sbm_function(df, snapshots_times, pi, seed)
    
    alive_matrix = interaction_counts + non_interaction_counts
    
    return interaction_counts, alive_matrix, df


def birth_death_function(lamb, mu, t_end, L, K, seed = 1):
    
    """
    Simulate a birth–death process.
    """

    d = {'id': np.arange(np.sum(L)), 'com': np.repeat(np.arange(K), L[0]), 't_birth': np.repeat(0, np.sum(L)),
     't_death': np.repeat(t_end, np.sum(L)), 'ancestor': np.repeat(-1, np.sum(L))}    
    df = pd.DataFrame(d)    
    tau = [0]
    b = []
    
    np.random.seed(seed)
    random.seed(seed)
    
    while tau[-1] < t_end :
        print(len(tau),  tau[-1])
        p = np.sum((lamb + mu)*L[-1])        
        delta = np.random.exponential(1/p)
        
        if (tau[-1] + delta > t_end):
            break
            
        tau = np.append(tau, tau[-1] + delta)  
        
        if len(df[df['t_death']>= t_end])>0:
            
            i = random.choice(df[df['t_death']>= t_end]['id'].tolist())
            k = int(df.loc[df['id'].eq(i), 'com'].iloc[0]) 
            
            if (np.random.uniform() < lamb[k]/(lamb[k] + mu[k])):
                #a node arrives
                new_L = copy(L[-1]) 
                new_L[k] = new_L[k] + 1            
                L = np.append(L, [new_L], axis=0)
                df.loc[len(df)] = [len(df), k , tau[-1], t_end, i] 
                b = np.append(b, 1)     
            
            else :
                # a node departs
                new_L = copy(L[-1] )
                new_L[k] = new_L[k] - 1
                L = np.append(L, [new_L], axis=0)
                df.loc[df['id'] == i, 't_death' ] = tau[-1] 
                b = np.append(b, -1)    
                
        else:
            break
            
    df['id'] = df['id'].astype(int)
    df['com'] = df['com'].astype(int)
    df['ancestor'] = df['ancestor'].astype(int)
    
    return df          

def discrete_graph_sbm_function(df, snapshots_times, pi, seed = 1):
    """
    Simulate a SBM according to the birth–death process.
    """
    np.random.seed(seed)
    
    N = len(df)
    graph = np.zeros((N, N), int)
    comp_graph = np.zeros((N, N), int)
    times = np.array(range(len(snapshots_times)))
    
    for i in np.arange(N):
        #print("i:", i)
        com_i, tb_i, td_i = df.loc[i][1:4]
        com_i = int(com_i)
        
        for j in np.arange(i+1, N):
                        
            com_j, tb_j, td_j = df.loc[j][1:4]  
            com_j = int(com_j)
            tb_ij = max(tb_i, tb_j)
            td_ij = min(td_i, td_j)
           
            if tb_ij < td_ij:
                
                Delta_ij = times[(snapshots_times >=  tb_ij) & (snapshots_times <=  td_ij)] 
                
                for l in Delta_ij:
                    
                    if (np.random.uniform() < pi[com_i][com_j]):
                                    
                        graph[i][j] += 1
                        graph[j][i] += 1
                        
                    else :
                    
                        comp_graph[i][j] += 1
                        comp_graph[j][i] += 1
                        
    return graph, comp_graph
