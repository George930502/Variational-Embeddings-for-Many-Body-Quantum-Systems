# Variational Mixed Quantum-Classical Simulation (Reproduction)

Reproduction of the paper:  
**"Variational Mixed Quantum-Classical Simulation with Conditional Wavefunctions"**  
by Stefano Barison, Filippo Vicentini, and Giuseppe Carleo (arXiv:2309.08666)

📄 [Original Paper (arXiv:2309.08666)](https://arxiv.org/abs/2309.08666)


## Overview

This project implements a faithful reproduction of the variational ansatz presented in the above paper, which expresses the total wavefunction of a quantum system with both **active** and **bath** orbitals.

The model captures entanglement between active and bath degrees of freedom using conditional neural networks, and estimates the ground state energy via **Variational Monte Carlo (VMC)** with **NetKet**.

## Key Formulations

This work is based on a hybrid quantum-classical decomposition of the wavefunction into **active** and **bath** degrees of freedom, with key equations from the paper summarized below.


### 1. Hamiltonian Decomposition

The full system Hamiltonian is decomposed as:

$$
\hat{H} = \hat{H}_A \otimes \mathbb{I}_B + \mathbb{I}_A \otimes \hat{H}_B + \hat{H}_{\text{int}} = \sum_j \hat{H}^{A,j} \otimes \hat{H}^{B,j}
$$


- $\hat{H}_A$, $\hat{H}_B$: act on **active** and **bath** subsystems  
- $\hat{H}_{\text{int}}$: interaction terms across subsystems

---

### 2. Conditional Wavefunction Ansatz

The wavefunction is decomposed as:

$$
|\Psi\rangle = \sum_{\sigma,\eta} \Psi(\sigma, \eta) |\sigma, \eta\rangle = \sum_{\sigma, \eta} \alpha(\sigma | \eta) \beta(\eta) |\sigma, \eta\rangle
$$

$\alpha(\sigma|\eta)$: learned conditional wavefunction (neural network)

$\beta(\eta)$: bath state amplitudes

---

### 3. Energy Expectation Value

The expected energy is:

$$
E = \frac{\langle \Psi | \hat{H} | \Psi \rangle}{\langle \Psi | \Psi \rangle}
  = \sum_{\sigma, \eta} p_\Psi(\sigma, \eta) E_{\text{loc}}(\sigma, \eta)
$$

Where:

$$
p_\Psi(\sigma, \eta) = \frac{|\Psi(\sigma, \eta)|^2}{\sum |\Psi(\sigma, \eta)|^2}
$$

$$
E_{\text{loc}}(\sigma, \eta) = \sum_{\sigma', \eta'} \frac{\Psi(\sigma', \eta')}{\Psi(\sigma, \eta)} H_{\sigma, \eta; \sigma', \eta'}
$$

---

### 4. Factorization Assumption

If we assume:

$$
\sum_{\sigma} |\alpha(\sigma|\eta)|^2 = 1 \quad \forall \eta
$$

Then the energy simplifies to:

$$
E = \sum_{\eta} p_\beta(\eta) \sum_{\sigma} |\alpha(\sigma|\eta)|^2 E_{\text{loc}}(\sigma, \eta)
$$

Where:

$$
p_\beta(\eta) = |\beta(\eta)|^2
$$

---

### Intuition

- Energy is computed as a **double average** over sampled bath configurations $\eta$ and active samples $\sigma$
- Conditional neural network models $\alpha(\sigma|\eta)$, and is optimized per bath sample