 #!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar  6 17:26:04 2023

@author: gabriela
"""

import numpy as np

def poisson_binomial_pmf(p):
    """
    Compute the probability mass function (PMF) of the Poisson Binomial distribution.
    Uses the recursive formula for exact computation.
    """
    n = len(p)
    pmf = np.zeros(n + 1)
    pmf[0] = 1  # P(X=0)
    
    for pi in p:
        pmf[1:] = pmf[1:] * (1 - pi) + pmf[:-1] * pi
        pmf[0] *= (1 - pi)
    
    return pmf

def compute_gamma_mar_l(df, delta, tau_l):
    
    N, K = delta.shape
    gamma_mar_l = np.zeros((K, N+1))
                      
    V_l = list(df[ (df['t_birth']< tau_l) & (df['t_death']>= tau_l)]['id'])
    n_values = list(range(len(V_l)+1))
            
    for k in range(K):
        
        gamma_mar_l[k][n_values] = poisson_binomial_pmf(delta[V_l, k])
       
    return gamma_mar_l

def compute_gamma_mar_init(df, tau, delta):
    
    M = len(tau) - 1
    N, K = delta.shape
   
    gamma_mar = np.zeros((M, K, N+1))
      
    for l in range(M):
       
       gamma_mar[l] = compute_gamma_mar_l(df, delta, tau[l+1])
       
    return gamma_mar


def compute_delta_gamma_init(df, K, delta_0, t0=0.0, smooth=1e-9):
    """
    Initialize community memebership probabilities for all individuals.
    
    - Individuals alive at t0: given by delta_0 (shape: N0 x K).
    - Newborns at time t: delta_i,k = sum_{alive before t} delta_j,k / (#alive before t),
      with a small smoothing to avoid exact zeros.

    Assumption: individuals present at t0 correspond to indices 0..N0-1.
    """
    tb = df["t_birth"].to_numpy(float)
    td = df["t_death"].to_numpy(float)
    N = len(df)

    N0 = delta_0.shape[0]
    delta = np.zeros((N, K), dtype=float)
    delta[:N0] = delta_0

    # alive set at t0
    alive = np.zeros(N, dtype=bool)
    alive[:N0] = True

    # expected alive mass per community among alive individuals
    n_k = delta[:N0].sum(axis=0)

    # process birth times > t0 in chronological order (grouped)
    birth_times = np.unique(tb[tb > t0])
    birth_times.sort()

    for t in birth_times:
        # remove those who died before (or at) time t
        dying = alive & (td <= t)
        if np.any(dying):
            n_k -= delta[dying].sum(axis=0)
            alive[dying] = False

        newborn = np.where(tb == t)[0]
        if newborn.size == 0:
            continue

        total_alive = n_k.sum()
        # p_k = (sum alive delta_k) / (#alive), smoothing to avoid zeros
        p = (n_k + smooth) / (total_alive + K * smooth)

        delta[newborn] = p
        alive[newborn] = True
        n_k += newborn.size * p

    return delta

def H_q(S1, S, df, beta, delta, pi, gamma, gamma_mar, b, lambda_est, mu_est, diff_tau):
    """
    Compute the ELBO.

    - term1: expected SBM log-likelihood using interaction counts (S1) and exposure (S)
    - term2: birth–death log-likelihood contribution (passed as input)
    - term3: birth-related combinatorial/transition term involving gamma and gamma_mar
    - term4: prior/mixing term at time 0 using beta (only for nodes alive at t=0)
    - term5: entropy term of delta for nodes alive at t=0
    - term6: entropy term of gamma 

    Returns
    -------
    H_q_value : float
    term1_icl_var : float
    """
    
    V_t0 = list(df[df['t_birth']<=0]['id'])
    times = np.array( list(range(len(b))))
    times_births = times[b== 1]
    times_deaths = times[b== -1]
    n_vals = np.arange(1, len(df)+1)
       
    # first term: sum_{i<j} sum_{k1, k2} delta(i, k1) delta(j, k2) sum_{l in Delta_ij} log(phi(e^l_ij, pi_k1k2))        
    term1 = 0.5 * np.sum(delta.dot(np.log((pi/(1-pi)))).dot(delta.transpose())*S1 + delta.dot(np.log(1-pi)).dot(delta.transpose())*S)
      
    # second term: sum_{l=1}^M sum_k sum_{n=1}^{N_l} -(lambda_k + mu_k) * Delta_l * n * gamma_mar(l-1, k, n)     
    E_lk = gamma_mar[ :, :, 1:] @ n_vals                         # E[N_{l,k}]     # (L, K)
    integ = diff_tau.dot(E_lk) 
    term2 = - np.sum((lambda_est + mu_est) * integ)
    
    # third term: sum_{l in T_B} sum_k sum_{n=1}^{N_l} gamma(l, k, n, n+1) gamma_mar(l-1, k, n) log(lambda_k*n)+gamma(l, k, n, n-1) gamma_mar(l-1, k, n) log(mu_k)
    log_lamb_n = np.log(lambda_est[:, None] * n_vals[None, :])
    term3 = np.sum((1-gamma[times_births, : , 1:])*gamma_mar[times_births, : , 1:] * log_lamb_n [None, :, :])  + np.sum((1-gamma[times_deaths, : , 1:])*gamma_mar[times_deaths, : , 1:] * np.log(mu_est[None, :, None]))             
    
    # fourth term: sum_{i in V_{t0}} sum_k delta(i, k) log(beta_k)
    term4 = np.sum(delta[V_t0].dot(np.log(beta)))
    
    # fifth term: sum_{i in V_{t0}} sum_k delta(i, k) log(delta(i, k))    
    term5 = delta[V_t0]*(np.log(delta[V_t0] ))
    term5 = np.sum(np.nan_to_num(term5, nan=0.0))
    
    #sixth term: sum_{l=1}^M sum_k sum_{n=1}^{N_l} gamma(l, k, n, n) gamma_mar(l-1, k, n) log(gamma(l, k, n, n))    
    safe_log_1 = np.zeros_like(gamma)
    mask = gamma > 0
    safe_log_1[mask] = np.log(gamma[mask])
    new_gamma_2 = 1 - gamma
    safe_log_2 = np.zeros_like(new_gamma_2)
    mask = new_gamma_2 > 0
    safe_log_2[mask] = np.log(new_gamma_2[mask])
    A = gamma*gamma_mar*safe_log_1+new_gamma_2*gamma_mar*safe_log_2
    #A[np.isnan(A)] = 0
    term6 =  np.sum(A)
    
    H_q_value = term1 + term2 + term3 + term4 - term5 - term6
    
    term1_icl_var = term1 + term2 + term3 + term4
    
    return H_q_value, term1_icl_var 


