#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 19 18:00:46 2025

@author: gabriela
"""

import numpy as np

class DynamicSBM_VEM:
    
    """
    Variational EM (VEM) inference for a dynamic Stochastic Block Model.

    """
    
    def __init__(self, interaction_counts, alive_matrix, delta_0, max_iters, tol):
        
        """
        Initialize the dynamic SBM VEM estimator.

        Parameters
        ----------
        interaction_counts :  array-like, shape (N, N)
            Pairwise interaction counts over the whole study interval. Entry (i, j) is the 
            total number of observed interactions between individuals i and j.
        alive_matrix : array-like, shape (N, N)
            Pairwise co-aliveness (exposure) matrix. Entry (i, j) is the number of
            observation times at which individuals i and j are simultaneously alive/active
            (i.e., the number of opportunities for them to interact).       
        delta_0 : array-like, shape (N0, K) or shape(N, K)
            Initial variational distribution over communities memberships.
        max_iters : int
            Maximum number of VEM iterations.
        tol : float
            Convergence tolerance for the stopping criterion.
        """
        
        # Number of communities
        self.K = delta_0.shape[1] 

        # Total number of individuals in the dataset
        self.N = delta_0.shape[0]     

        self.delta_0 = delta_0        
        self.max_iters = max_iters
        self.tol = tol          
        self.S1 = interaction_counts
        self.S = alive_matrix     
        self.M = np.sum(self.S)         
        
        # Pre-computed penalty term used in model selection 
        self.pen = 0.5*(0.5 * self.K * (self.K+1) * np.log(max(self.M/2, 1.0)) + (self.K - 1) * np.log(self.N))
                     
        
    def expectation_step(self, delta, pi, beta):
        """
        Perform the VEM E-step for the dynamic SBM.
        """
    
        #to upgrade delta
        A = (np.log((pi/(1-pi)))).dot(delta.transpose()).dot(self.S1) +(np.log(1-pi)).dot(delta.transpose()).dot(self.S)
        At = A.transpose()
        maxi = np.max(At, axis=1)      
        #maxi = np.mean(At, axis=1)      
        res = At - maxi[:, np.newaxis]        
        new_delta = np.exp(res)*beta
        new_delta = new_delta.transpose()*(1/ np.sum(new_delta, axis = 1)) # Normalize 
        new_delta = new_delta.transpose()
                                          
        return  new_delta
            
    def maximization_step(self, delta):
        
        """
        Perform the VEM M-step for the dynamic SBM.
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
            
        beta = delta[0:self.N].mean(axis = 0)
        
        return  pi_new, beta
    
    def H_q(self, delta, beta, pi):
            
        # 1st term: sum_{i<j} sum_{k1, k2} delta(i, k1) delta(j, k2) sum_{l in Delta_ij} log(phi(e^l_ij, pi_k1k2))   
        term1 = 0.5 * np.sum(delta.dot(np.log((pi/(1-pi)))).dot(delta.transpose())*self.S1 + delta.dot(np.log(1-pi)).dot(delta.transpose())*self.S)
        #print('term1:',term1)
        
        #2nd term: sum_{i in V_{t0}} sum_k delta(i, k) log(beta_k)
        term2 = np.sum(delta.dot(np.log(beta)))
            
        #3rd term: sum_{i in V_{t0}} sum_k delta(i, k) log(delta(i, k))
        d0 = delta
        d0_safe = np.where(d0 == 0, 1.0, d0)   # 0 -> 1 so that log(1)=0
        term3 = np.sum(d0 * np.log(d0_safe))
                
        H_q_value = term1 + term2 - term3 
                
        icl_var = term1 + term2 - self.pen

        return H_q_value, icl_var
    
    def fit(self ):
        """
        Fit the model using Variational Expectation-Maximization (VEM).
        """
             
        # Initialization of delta        
        delta = self.delta_0
          
        # Initialization of pi and beta 
        pi, beta = self.maximization_step( delta)
        
        elbo_values = []
                
        for i in range(self.max_iters):
            
            # E-step: 
            delta = self.expectation_step( delta, pi, beta)
            
            # M-step: 
            pi, beta = self.maximization_step( delta)
            
            # to compute the ELBO and the ICL variational
            current_elbo, icl_var = self.H_q(delta, beta, pi)
            elbo_values.append(current_elbo)
            
            # Convergence criterion
            if i > 0 and np.abs(elbo_values[-1] - elbo_values[-2]) < self.tol:
                print("Convergence reached")
                break
        
        return pi, beta, delta, elbo_values, icl_var

