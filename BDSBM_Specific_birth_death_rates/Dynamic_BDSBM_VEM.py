#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 19 18:00:46 2025

@author: gabriela
"""

import numpy as np
from scipy.optimize import fsolve
from .utils import compute_delta_gamma_init, H_q, poisson_binomial_pmf, compute_gamma_mar_init

class DynamicBDSBM_VEM:
    
    """
    Variational EM (VEM) inference for a dynamic Birth–Death Stochastic Block Model (BD-SBM).

    This class implements a variational EM procedure to infer latent community
    memberships and model parameters from dynamic interaction data where the
    node set evolves over time according to a birth–death process.

     The model is observed through:
    - `interaction_counts` (S1): a (N × N) interaction-count matrix where
      S1[i, j] is the total number of observed interactions between individuals i and j
      over the whole study interval.
    - `alive_matrix` (S): a (N × N) exposure / co-aliveness matrix where
      S[i, j] is the number of observation times at which individuals i and j are
      simultaneously alive/active (i.e., the number of opportunities for them to interact).
    - `df_births_deaths`: individual birth and death times defining the evolving node set.

    The VEM algorithm estimates:
    - birth–death parameters (lambda, mu),
    - dynamic SBM parameters (pi, beta),    
    - variational probabilities (delta) and auxiliary probabilities  (gamma, gamma_mar),
    and monitors convergence using the ELBO.
    """
    
    def __init__(self, interaction_counts, alive_matrix, df_births_deaths, delta_0, max_iters, tol):
        
        """
        Initialize the dynamic BD-SBM VEM estimator.

        Parameters
        ----------
        interaction_counts :  array-like, shape (N, N)
            Pairwise interaction counts over the whole study interval. Entry (i, j) is the 
            total number of observed interactions between individuals i and j.
        alive_matrix : array-like, shape (N, N)
            Pairwise co-aliveness (exposure) matrix. Entry (i, j) is the number of
            observation times at which individuals i and j are simultaneously alive/active
            (i.e., the number of opportunities for them to interact).
        df_births_deaths : pandas.DataFrame
            Per-individual information including at least columns:
            - 't_birth': birth time of the individual
            - 't_death': death time of the individual
        delta_0 : array-like, shape (N0, K) or shape(N, K)
            Initial variational distribution over communities (e.g., responsibilities).
        max_iters : int
            Maximum number of VEM iterations.
        tol : float
            Convergence tolerance for the stopping criterion.
        """
        
        # Number of communities
        self.K = delta_0.shape[1] 

        # Birth/death dataframe        
        self.df = df_births_deaths 
        
        # Number of individuals present at time 0 (t_birth <= 0)
        self.N0 = len(self.df[self.df['t_birth']<=0]) 
        
        # Total number of individuals in the dataset
        self.N = len(df_births_deaths)    

        t_end = np.max(df_births_deaths['t_death'])
        self.index_deaths = df_births_deaths[df_births_deaths['t_death']< t_end]['id']
        
        # Sorted births and deaths times     
        self.tau = np.sort(np.unique(np.concatenate((self.df['t_birth'], self.df['t_death']))))[:-1]     
        self.diff_tau = np.diff(self.tau)
        
        # Event-type vector along `tau`: +1 birth, -1 death
        b = np.full(len(self.tau), -1, dtype=int)
        birth_times = self.df["t_birth"].to_numpy(dtype=float)
        mask = np.isin(self.tau, birth_times)
        b[mask] = 1
        b = b[1:]
        self.b = b                              
        
        # Number of alive individuals after each event time (aligned with `b`)            
        self.N_l = self.N0 + np.cumsum(np.concatenate(([0], b))) 
        
        self.delta_0 = delta_0        
        self.max_iters = max_iters
        self.tol = tol          
        self.S1 = interaction_counts
        self.S = alive_matrix     
        self.M = np.sum(self.S)
        
        # Indices of event times (aligned with b) 
        self.times = np.array( list(range(len(self.b))))      
        
        # Indices corresponding to birth events
        self.times_births = self.times[self.b== 1]
        
        # Pre-computed penalty term used in model selection 
        self.pen = 0.5*(0.5 * self.K * (self.K+1) * np.log(max(self.M/2, 1.0)) + (self.K - 1) * np.log(self.N0) + (self.K) * (np.log(len(self.times_births)) + np.log(len(b) - len(self.times_births))))              
        #self.pen = 0.5*(0.5 * self.K * (self.K+1) * np.log(max(self.M/2, 1.0)) + (self.K - 1) * np.log(self.N0) + 2*(self.K) * (np.sum(self.N_l)))              
                                 
        
    def expectation_step(self, delta, pi, beta, lambda_est, mu_est):
        """
        Perform the VEM E-step for the dynamic BD-SBM.
        
        This E-step updates the variational distributions given the current model
        parameters. It (i) updates the node probabilities `delta`, and (ii) propagates 
        variational quantities (`gamma`, `gamma_mar`) along the birth–death event 
        timeline to account for the varying number of alive individuals.
    
        Parameters
        ----------
        delta : array-like, shape (N, K)
            Current community membership probabilities matrix.
            Row i corresponds to individual i and sums to 1 across communities.
        pi : array-like, shape (K, K)
            Current SBM connection probability matrix.
        beta : array-like, shape (K,) 
            Current vector of community probabilities at time t_0.
        lambda_est : array-like, shape (K,) 
            Current vector of community bith rates.
        mu_est : array-like, shape (K,) 
            Current vector of community death rates.
    
        Returns
        -------
        new_delta : array-like, shape (N, K)
            Updated of community membership probabilities matrix.
        gamma : array-like, shape (T, K, N+1)
            Variational transition probabilities of the the community-sizes over the 
            event timeline. For each event index l, community k, and count n,
            `gamma[l, k, n]` denotes the transition probability to move from n to n 
            individuals in community k at time tau_l+1. 
            Implementation note: the original quantity can be viewed as γ(l, k, n, n),
            which is stored here as a 3D tensor for convenience. The indexing is shifted:
            since there is no γ at the initial time, `gamma[l]` corresponds to the update
            from event time τ_l to τ_{l+1}. The first column (n = 0) is fixed to 1 for all
            l and k.
        gamma_mar : array-like, shape (T, K, N+1)
            Marginal distributions of community sizes over the event timeline.
            `gamma_mar[l, k, n]` is the probability that, at event index l, there are n
            alive individuals in community k. It is initialized at time 0 using a
            Poisson–binomial pmf and then updated recursively at each birth/death event.
        """
        
        mu_matrix = np.ones((self.N, self.K))
        mu_matrix[self.index_deaths, :] = mu_est[None, :]        
    
        V = list(range(self.N + 1))
        V_t0 = list(range(self.N0))
        zeros = np.zeros((self.K, 1))
        
        #to upgrade delta for i in V_t_0
        A = (np.log((pi/(1-pi)))).dot(delta.transpose()).dot(self.S1) +(np.log(1-pi)).dot(delta.transpose()).dot(self.S)
        At = A.transpose()
        maxi = np.max(At, axis=1)      
        #maxi = np.mean(At, axis=1)      
        res = At - maxi[:, np.newaxis]        
        new_delta = np.exp(res)*beta
        new_delta[V_t0] = new_delta[V_t0] * mu_matrix[V_t0]        
        new_delta = new_delta.transpose()*(1/ np.sum(new_delta, axis = 1)) # Normalize 
        new_delta = new_delta.transpose()
        new_delta_0 = new_delta[V_t0]                 
        
        vector_log = np.log(np.arange(1, self.N))  # Log of the numbers from 1 to N
       
        At_ = res[len(V_t0):, :]
        B = At_[:, :, np.newaxis] + vector_log    
        new_B = np.exp(B)
        
        # to initialize gamma_mar
        gamma_mar = np.zeros((len(self.b), self.K, self.N+1))   
        
        # to compute gamma_mar at time 0
        for k in range(self.K):            
            gamma_mar[0][k][V_t0 + [len(V_t0)]] = poisson_binomial_pmf(new_delta_0[:, k])
        
        # to initialize gamma
        gamma = np.zeros((len(self.b), self.K, self.N))
        ones_column = np.ones((gamma.shape[0], gamma.shape[1], 1))
        gamma = np.concatenate((ones_column, gamma), axis=2)
    
        idx = 0
        zeros = np.zeros((self.K, 1))
        error = 1e-10  
               
        #to upgrade gamma, gamma_mar, delta for each tau_l
        for l in range(len(self.b)-1):
           
            v = 1 - V / self.N_l[l]
            v[v<0] = 0    
           
            if l in self.times_births:
                
                alpha0 = np.sqrt(1/np.max(new_B[idx]))
                lamb_mu = lambda_est*mu_matrix[idx]
                
                if alpha0 == 0:
                   alpha0 = 1e-16
                   
                def system(alpha):
                   
                    gamma[l, :, 1:self.N_l[l]] = 1 / (1 + ((alpha)**2)*lamb_mu[:, None]*new_B[idx, :, 0:self.N_l[l]-1])                    
                    eq = np.sum(gamma[l] * gamma_mar[l]) - self.K + 1 
            
                    return eq

                alpha1 = fsolve(system, alpha0)      
                residual = system(alpha1)
                
                if abs(residual) > error:
                   
                   multipliers=(0.1, 10, 100, 1000)
                   best_alpha = alpha1
                   best_residual = residual
                    
                   for m in multipliers:
                      
                      new_alpha0 = alpha0 * m
                      new_alpha = fsolve(system, new_alpha0)      
                      residual = system(new_alpha)
                      
                      if np.isfinite(new_alpha) and np.isfinite(residual) and abs(residual) < abs(best_residual):
                         best_alpha, best_residual = new_alpha, residual
                   
                   # final sync : make gamma consistent with the chosen alpha
                   gamma[l, :, 1:self.N_l[l]] = 1 / (1 + ((best_alpha)**2)*new_B[idx, :, 0:self.N_l[l]-1])                  
                
                idx = idx + 1
                i = self.df[self.df['t_birth'] == self.tau[l+1]]['id']
                new_delta[i] = np.sum((1- gamma[l])*gamma_mar[l], axis=1)
                        
                if l< len(self.b)-1:     
                   m1 = gamma[l]*gamma_mar[l]
                   m2 = np.hstack((zeros, (1-gamma[l])*gamma_mar[l],zeros))    
                   n = m1 + m2[:, int((1-self.b[l])):self.N + 1 + int((1-self.b[l]))]                
                   gamma_mar[l + 1] = n
                                     
            else: 
                        
                i = self.df[self.df['t_death'] == self.tau[l+1]]['id'] 
                                
                r = np.arange(1, self.N_l[l])
                               
                for k in range(self.K):
                    
                    if new_delta[i,k] == 0:
                        
                        gamma[l, k, 1:self.N_l[l]] = 1
                  
                    else :
                       
                        alpha0 = 1
                                                
                        def system(alpha):
                   
                            gamma[l, k, 1:self.N_l[l]] = 1 / (1 + ((alpha)**2)*r)
                            eq = np.sum(gamma[l,k,:] * gamma_mar[l,k,:]) - 1 + new_delta[i,k] 
            
                            return eq

                        alpha1 = fsolve(system, alpha0)                   
                        residual = system(alpha1)
                        
                        if abs(residual) > error:
                           
                           multipliers=(0.1, 10, 100, 1000, 10000, 100000)
                           best_alpha = alpha1
                           best_residual = residual
                           
                           for m in multipliers:
                              
                              new_alpha0 = alpha0 * m
                              new_alpha = fsolve(system, new_alpha0)      
                              residual = system(new_alpha)
                              
                              if np.isfinite(new_alpha) and np.isfinite(residual) and abs(residual) < abs(best_residual):
                                 best_alpha, best_residual = new_alpha, residual
                          
                           # final sync : make gamma consistent with the chosen alpha
                           gamma[l, k, 1:self.N_l[l]] = 1 / (1 + ((best_alpha)**2)*r)        
                                        
                if l< len(self.b)-1:  
                                        
                    m1 = gamma[l]*gamma_mar[l]
                    m2 = np.hstack((zeros, (1-gamma[l])*gamma_mar[l],zeros))    
                    n = m1 + m2[:, int((1-self.b[l])):self.N + 1 + int((1-self.b[l]))]                
                    gamma_mar[l + 1] = n
                               
                    if np.any(gamma_mar[l+1] < 0):
                       A = np.asarray(gamma_mar[l+1], dtype=float)
                       mask = A < 0 
                       masked = np.where(mask, A, -np.inf)
                       idx_flat = np.argmax(masked)              
                       pos = np.unravel_index(idx_flat, A.shape) 
                       val = A[pos]
                       
                       if abs(val) < error:
                          gamma_mar[l+1] = np.maximum(gamma_mar[l+1], 0, out=gamma_mar[l+1])                 
                
        return  new_delta, gamma, gamma_mar  
            
    def maximization_step(self, delta, gamma_mar):
        
        """
        Perform the VEM M-step for the BD-SBM.
        """                
        eps=1e-20
        
        num_pi = delta.T @ self.S1 @ delta
        den_pi = delta.T @ self.S  @ delta
        pi = num_pi / den_pi
    
        if np.any(den_pi == 0):
            
            print("Warning: den_pi has zeros in", np.argwhere(den_pi == 0).tolist())
            pi_new = pi.copy()  # fallback: keep previous pi
            mask = den_pi > 0
            pi_new[mask] = num_pi[mask] / den_pi[mask]
        
            # stability for logs later
            pi_new = np.clip(pi_new, eps, 1 - eps)
            
        else :
            pi_new = pi
            
        beta = delta[0:self.N0].mean(axis = 0)
        
        n_vals = np.arange(self.N + 1, dtype=np.float64)       # (N_max,)
        
        E_lk = gamma_mar  @ n_vals                         # E[N_{l,k}]     # (L, K)
        denom = self.diff_tau.dot(E_lk)                    # (K,)
        
        lambda_est = delta[self.N0:].sum(axis = 0)/denom
        
        mu_est = delta[self.index_deaths].sum(axis = 0)/denom
        
        return  pi_new, beta, lambda_est, mu_est
    
    def ICL_effective(self, delta, gamma_mar):
        """
        Compute the Integrated Completed Likelihood (ICL) criterion.
        """
        
        eps = 1e-30
        N_nodes, K = delta.shape
        V_t0 = np.asarray(self.df.loc[self.df['t_birth'] <= 0, 'id'], dtype=int)
        
        idx = np.argmax(delta, axis=1)       # MAP by individual
        Z_full = np.zeros_like(delta, dtype=int)
        Z_full[np.arange(N_nodes), idx] = 1  # Z_full: (N, K)
    
        # number of individuals by communities
        n_k_full = Z_full.sum(axis=0)        # (K,)
    
        # non empty communities
        mask = n_k_full > 0
        K_effective = int(mask.sum())        
    
        # Z for non empty communities
        Z = Z_full[:, mask]                  # (N, K_eff)

        num_pi = Z.T @ self.S1 @ Z   
        den_pi = Z.T @ self.S  @ Z   
           
        pi_hat = np.divide(num_pi,
                       den_pi,
                       out=np.zeros_like(num_pi, dtype=float),
                       where=den_pi > 0)

        # to avoid log(0) 
        pi_hat = np.clip(pi_hat, eps, 1.0 - eps)
            
        n_k = Z.sum(axis=0)               
        beta = n_k / float(N_nodes)       
        
        logit_pi = np.log(pi_hat / (1.0 - pi_hat))
        log1m_pi = np.log(1.0 - pi_hat)
        
        # first term: 
        term1 = 0.5 * np.sum( (Z @ logit_pi @ Z.T) * self.S1 + (Z @ log1m_pi @ Z.T) *self.S)
                
        # second term:
        n_idx = np.argmax(gamma_mar, axis=2)
        L = np.zeros_like(gamma_mar, dtype=np.uint8)
        t_idx = np.arange(gamma_mar.shape[0])[:, None]       # shape (T,1)
        k_idx = np.arange(gamma_mar.shape[1])[None, :]       # shape (1,N)        
        L[t_idx, k_idx, n_idx] = 1                           # to put 1 in (t, k*, n)
        t_births = self.times[0:-1][self.b[0:-1]== 1]
        t_deaths = self.times[0:-1][self.b[0:-1]== -1]
        
        n_vals = np.arange(0, N_nodes + 1)       # (N_max,)        
        E_lk = L  @ n_vals                       # E[N_{l,k}]     # (L, K)
        denom = self.diff_tau.dot(E_lk)          # (K,)
        
        lambda_est = Z_full[len(V_t0):].sum(axis = 0)/denom        
        mu_est = Z_full[self.index_deaths].sum(axis = 0)/denom  
     
        arg_lamb = lambda_est[:, None] * n_vals[None, 1:self.N-1]
        arg_lamb_safe = arg_lamb.copy()
        arg_lamb_safe[arg_lamb_safe == 0] = 1.0
        arg_lamb_safe[np.isnan(arg_lamb_safe)] = 1.0
        log_lamb_n = np.log(arg_lamb_safe)    
        term2_1 = np.sum((L[t_births + 1, : , 2:self.N])*L[t_births, : , 1:self.N-1]*log_lamb_n [None, :, :])
       
        arg_mu = mu_est[:, None]
        arg_mu_safe = arg_mu.copy()
        arg_mu_safe[arg_mu_safe == 0] = 1.0
        arg_mu_safe[np.isnan(arg_mu_safe)] = 1.0
        log_mu_n = np.log(arg_mu_safe)    
        term2_2 = np.sum((L[t_deaths + 1, : , 0:self.N-2])*L[t_deaths, : , 1:self.N-1]*log_mu_n [None, :, :])
       
        # third term
        term3 = np.sum(Z[V_t0] @ np.log(beta))  
       
        pen = 0.25 * K_effective * (K_effective + 1) * np.log(max(self.M/2, 1.0)) + 0.5*(K_effective - 1) * np.log(self.N0) + 0.5*K_effective * (np.log(len(self.times_births)) + np.log(len(self.b) - len(self.times_births)))  

        ICL_value = term1 + term2_1 + term2_2 + term3 - pen
        
        return ICL_value
        
    def fit(self ):
        """
        Fit the model using Variational Expectation-Maximization (VEM).
        """
             
        if self.delta_0.shape[0] == self.N0:    
            delta = compute_delta_gamma_init(self.df, self.K, self.delta_0)
                                    
        elif self.delta_0.shape[0] == self.N: 
            delta = self.delta_0
            self.delta_0 = self.delta_0[0:self.N0]
        
        # Initialization of gamma_mar
        gamma_mar = compute_gamma_mar_init(self.df, self.tau, delta)
        
        # Initialization of pi and beta 
        pi, beta, lambda_est, mu_est = self.maximization_step( delta, gamma_mar)
        
        elbo_values = []
        icl_var_values = []
        
        for i in range(self.max_iters):
            
            # E-step: 
            delta, gamma, gamma_mar = self.expectation_step( delta, pi, beta, lambda_est, mu_est)
            
            # M-step: 
            pi, beta, lambda_est, mu_est = self.maximization_step( delta, gamma_mar)
            
            # to compute the ELBO and the ICL variational
            current_elbo, icl_var_term = H_q(self.S1, self.S, self.df, beta, delta, pi, gamma, gamma_mar, self.b, lambda_est, mu_est, self.diff_tau)
            elbo_values.append(current_elbo)
            icl_var_values.append(icl_var_term - self.pen) 
                        
            print()
            print(f"Iteration {i+1}, ELBO: {current_elbo}")
        
            #Convergence criterion
            if i > 0 and np.abs(elbo_values[-1] - elbo_values[-2]) < self.tol:
                print("Convergence reached")
                break
        
        icl = self.ICL_effective( delta, gamma_mar)
        
        return pi, beta, lambda_est, mu_est, delta, gamma, gamma_mar, elbo_values, icl, icl_var_values

