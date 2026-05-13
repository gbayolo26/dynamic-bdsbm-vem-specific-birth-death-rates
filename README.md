# dynamic-bdsbm-vem
## Variational EM inference for a dynamic Birth–Death Stochastic Block Model

This repository provides a **Variational EM (VEM)** implementation for a **dynamic Birth–Death Stochastic Block Model (BD-SBM)**. The model is designed for networks in which individuals **arrive and depart over time** according to a **birth–death process**.

This repository provides tools to:

1) Generate or load data (`interaction_counts`, `alive_matrix`, and birth/death times),  
2) Fit the dynamic BD-SBM using VEM (including a variant for **sparse** interaction data).

---

### Main components

- Data generation (birth–death process + SBM interactions)  
  ([code here](./BDSBM/data_generation.py))

- VEM inference for the dynamic BD-SBM (dense interaction counts)  
  ([code here](./BDSBM/Dynamic_BDSBM_VEM.py))

- VEM inference for the dynamic BD-SBM (sparse CSR interaction counts)  
  ([code here](./BDSBM/Dynamic_BDSBM_VEM_sparse.py))

- Utility functions (ELBO, ICL computation, Poisson–binomial distribution, etc.)  
  ([code here](./BDSBM/utils.py))

---

### More information

- Paper / preprint: ([link here](https://hal.science/hal-05430728))

---

### Usage

The following notebook contains an example showing how to run the model:

- **Dynamic Birth–Death SBM (VEM) notebook**  
  ([notebook here](https://github.com/gbayolo26/dynamic-bdsbm-vem/blob/main/Dynamic_Birth-Death_SBM_VEM.ipynb))

A complete end-to-end example (data generation → model fitting → model selection → plots) is provided.

---

### Simulation setting (brief)

We evaluate the method through simulations where:

- a birth–death process generates individual lifetimes and community memberships,
- an SBM with connection matrix `pi` generates interactions at observation times,
- VEM jointly estimates `(pi, beta, lambda, mu)` and the variational distributions `(delta, gamma, gamma_mar)`,
- model selection can be performed using ICL.

---

### Maintainers
Gabriela Bayolo Soler (gabriela.bayolo-soler@utc.fr)

<img src="https://github.com/gbayolo26/risk_estimation/assets/79975920/eedd8e6e-6cea-4327-bef3-54fe0256ff06" width="180" height="50">

---

### Licence
[Apache-2.0 licence](https://github.com/gbayolo26/risk_estimation/blob/main/LICENSE)
