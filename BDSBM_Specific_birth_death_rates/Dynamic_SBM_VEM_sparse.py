#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 19 18:00:46 2025

@author: gabriela
"""

import numpy as np

class DynamicSBM_VEM_sparse:
    
    """
    Variational EM (VEM) inference for a dynamic Stochastic Block Model
    """
    
    def __init__(self, sparse_adjacency_matrix, alive_matrix, delta_0, max_iters, tol):
        
        """
        Initialize the dynamic SBM VEM estimator.
        """
        
        # Number of communities
        self.K = delta_0.shape[1] 
        
        # Total number of individuals in the dataset
        self.N = delta_0.shape[0]   
        
        self.delta_0 = delta_0        
        self.max_iters = max_iters
        self.tol = tol          
        self.S1 = sparse_adjacency_matrix.tocsr()
        self.S = alive_matrix     
        self.M = np.sum(self.S)
            
        # Pre-computed penalty term used in model selection 
        self.pen = 0.5*(0.5 * self.K * (self.K+1) * np.log(max(self.M/2, 1.0)) + (self.K - 1) * np.log(self.N))             
        
        
    def expectation_step(self, delta, pi, beta):
        """
        E-Step: Perform the VEM E-step for the dynamic BD-SBM.
        """
                
        #to upgrade delta for i in V_t_0
        L1 = np.log(pi/(1-pi)).astype(np.float64)     # (K,K)
        L2 = np.log1p(-pi).astype(np.float64)         # (K,K)
        Delta = np.asarray(delta, dtype=np.float64)   # (N,K)
          
        #to compute S @ Delta o Delta.T @ S
        DT_S1 = Delta.T @ self.S1                             # (K, N)   denso@sparse -> denso       
        A  = (L1 @ DT_S1) + (L2.dot(Delta.T).dot(self.S))     # (K,K)@(K,N) -> (K,N)
        At = A.T                                              # (N,K)

        #to upgrade delta for i in V_t_0
        maxi = np.max(At, axis=1)      
        #maxi = np.mean(At, axis=1)      
        res = At - maxi[:, np.newaxis]    
        new_delta = np.exp(res)*beta 
        new_delta = new_delta.transpose()*(1/ np.sum(new_delta, axis = 1))        
        new_delta = new_delta.transpose()
        
        return  new_delta
    
    def maximization_step(self, delta):
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
           
        beta = delta.mean(axis = 0)
       
        return  pi_new, beta
        
    
    def H_q(self, beta, delta, pi):

        """
        Compute the ELBO.
        """                
       
        Delta  = delta.astype(np.float64, copy=False)
        L1 = np.log(pi).astype(np.float64)
        
        X1 = Delta @ L1    
        S1Delta = self.S1 @ Delta
    
        t1 = np.einsum('ik,ik->', X1, S1Delta)
        
        t2 = delta.dot(np.log(1-pi)).dot(delta.transpose())*self.S
        
        term1 = 0.5 * np.sum(t1 + t2)
       
        term2 = np.sum(delta.dot(np.log(beta)))
        
        term3 = np.sum(delta*(np.log(delta)))
               
        H_q_value = term1 + term2 - term3 
        
        term1_icl_var = term1 + term2 - self.pen
        
        return H_q_value, term1_icl_var 

    
    def ICL(self, delta, pi):
        """
        Compute the Integrated Completed Likelihood (ICL) criterion.
        """
        idx = np.argmax(delta, axis=1)
        z = np.zeros_like(delta, dtype=int)
        z[np.arange(delta.shape[0]), idx] = 1         
        num_pi = z.T @ (self.S1 @ z)
        den_pi = z.T.dot(self.S).dot(z)                
        pi = num_pi/den_pi
        pi[np.isnan(pi)] = 0
                   
        beta = z.mean(axis = 0) 
           
        # first term: 
        Z = z.astype(np.float64, copy=False)
        L1 = np.log(pi).astype(np.float64)           
        X1 = Z @ L1  
        S1Z = self.S1 @ Z
        t1 = np.einsum('ik,ik->', X1, S1Z)
            
        t2 = z.dot(np.log(1-pi)).dot(z.transpose())*self.S    
        term1 = 0.5 * np.sum(t1 + t2)
                
        #2nd term
        term2 = np.sum(z.dot(np.log(beta)))
       
        ICL_value = term1 + term2 - self.pen
        
        return ICL_value

    
    def ICL_effective(self, delta, pi):
        """
        Compute the Integrated Completed Likelihood (ICL) criterion.
        """
        
        eps = 1e-30
        N_nodes, K = delta.shape
        
        idx = np.argmax(delta, axis=1)       # MAP by individual
        Z_full = np.zeros_like(delta, dtype=int)
        Z_full[np.arange(N_nodes), idx] = 1  # Z_full: (N, K)
    
        # number of individuals by community
        n_k_full = Z_full.sum(axis=0)        # (K,)
    
        # non empty communities
        mask = n_k_full > 0
        K_effective = int(mask.sum())        
    
        # Z reduced to non empty communities
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
                
        
        # 2nd term
        term2 = np.sum(Z @ np.log(beta))  
       
        pen = 0.25 * K_effective * (K_effective + 1) * np.log(max(self.M/2, 1.0)) + 0.5*(K_effective - 1) * np.log(self.N)  

        ICL_value = term1 + term2 - pen
        
        return ICL_value

    
    def fit(self ):
        """
        Fit the model using Variational Expectation-Maximization (VEM).
        """
            
        delta = self.delta_0
            
        # Initialization of pi and beta 
        pi, beta = self.maximization_step( delta)
                
        elbo_values = []
        icl_var_values = []
        
        for i in range(self.max_iters):
            
            # E-step:
            delta = self.expectation_step( delta, pi, beta)
         
            # M-step: 
            pi, beta = self.maximization_step( delta)
        
            # to compute the ELBO and the ICL variational
            current_elbo,icl_var_term = self.H_q( beta, delta, pi)
            elbo_values.append(current_elbo)
            icl_var_values.append(icl_var_term - self.pen) 
                        
            print()
            print(f"Iteración {i+1}, ELBO: {current_elbo}")
        
            # Convergence criterion
            if i > 0 and np.abs(elbo_values[-1] - elbo_values[-2]) < self.tol:
                print("Convergencia alcanzada")
                break
                    
        icl = self.ICL_effective(delta, pi)  
        
        return pi, beta, delta, elbo_values, icl, icl_var_values


