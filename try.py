import numpy as np
import jax
import jax.numpy as jnp
import flax.linen as nn
from flax.core.frozen_dict import unfreeze
import netket as nk
from netket.operator import LocalOperator
from netket.operator.spin import sigmax, sigmaz
from netket.optimizer import Sgd, SR
from netket.exact import lanczos_ed
from netket.driver import VMC
from netket.vqs import MCState
from itertools import product
import matplotlib.pyplot as plt
from typing import Any

jax.config.update("jax_enable_x64", True)   # jax-64 bits mode

# =====================
# Config
# =====================
n_active, n_bath = 3, 8
ranges_to_test = [1, 2]
n_samples_eta = 200
vmc_iterations = 3000
learning_rate = 0.01

J_A = 10.0
J_B = 1e-2
J_int = 2
H_field = 1.0

# =====================
# Utils
# =====================
def all_bath_configs(n):
    return np.array(list(product([1, -1], repeat=n)))  # 2^n

def jastrow_weights(etas, J1, J2):
    # Boundary condition is periodic
    # Whether spin direction is the same or not
    # Jastrow 型波函數的典型結構之一，用來加入對二體交互作用（pairwise interaction）的建模
    # return np.array([np.exp(2 * J * np.sum(e * np.roll(e, -1))) for e in etas])
    return np.array([np.exp(2 * J1 * np.sum(e * np.roll(e, -1)) + (2 * J2 * np.sum(e * np.roll(e, -2)))) for e in etas])

def sample_bath(n_bath, J1, J2, n_samples):
    etas = all_bath_configs(n_bath)
    ws = jastrow_weights(etas, J1, J2)
    ps = ws / ws.sum()
    idx = np.random.choice(len(etas), size=n_samples, p=ps)
    counts = np.bincount(idx, minlength=len(etas))
    # print(counts)
    return [(etas[i], counts[i] / n_samples)
            for i in range(len(etas)) if counts[i] > 0]

def get_local_eta(eta, rng):
    return eta[:rng]

def sigma_to_index(sigma):
    bits = (sigma > 0).astype(jnp.int32)
    power = 2 ** jnp.arange(bits.shape[-1], dtype=jnp.int32)
    return jnp.sum(bits * power, axis=-1)

# =====================
# NN Modules
# =====================
class BackflowNN(nn.Module):
    n_input: int
    n_output: int
    @nn.compact
    def __call__(self, eta_local): 
        x = eta_local.astype(jnp.float64)
        x = nn.Dense(4, dtype=jnp.float64)(x)   # W1: (n_input × m), b1: (m)  
        x = nn.tanh(x)  
        x = nn.Dense(4, dtype=jnp.float64)(x)   # W2: (m × m), b2: (m)
        x = nn.tanh(x)
        return nn.Dense(self.n_output, dtype=jnp.float64)(x)   # W3: (m × output), b3: (output)

class StateVector(nn.Module):
    hilbert: Any
    @nn.compact
    def __call__(self, sigma):
        n = self.hilbert.n_states  
        amps = self.param("amps", nn.initializers.normal(0.1), (n,))
        amps = amps.astype(jnp.float64)
        idx = sigma_to_index(sigma)
        return amps[idx]

class MixedAnsatz(nn.Module):
    bf_input: int
    bf_output: int
    hilbert: Any

    def setup(self):
        self.bf = BackflowNN(self.bf_input, self.bf_output)
        self.sv = StateVector(self.hilbert)

    def __call__(self, sigma):
        f = self.bf(current_eta_local)
        a0 = self.sv(sigma)
        idx = sigma_to_index(sigma)
        return a0 + f[idx]

# =====================
# Hamiltonian
# =====================
def make_active_hamiltonian():
    hilb = nk.hilbert.Spin(s=0.5, N=n_active)
    H = LocalOperator(hilb)
    for i in range(n_active - 1):
        H += -J_A * sigmaz(hilb, i) * sigmaz(hilb, i + 1)
    for i in range(n_active):
        H += -H_field * sigmax(hilb, i)
    return H, hilb

def make_full_tfim_hamiltonian(J_couplings, h=1.0):
    N = len(J_couplings) + 1  # 總 spin 數 = J_i 數 + 1
    hilbert = nk.hilbert.Spin(s=0.5, N=N)
    H = LocalOperator(hilbert)
    for i, J in enumerate(J_couplings):
        H += -J * sigmaz(hilbert, i) * sigmaz(hilbert, i + 1)
    for i in range(N):
        H += -h * sigmax(hilbert, i)
    return H, hilbert

# =====================
# Exact Ground State Energy
# =====================
def get_exact_ground_state_energy(H):
    eigenvalues = lanczos_ed(H, k=1)
    return eigenvalues[0]

# =====================
# Run Backflow
# =====================
def run_backflow(rng):
    J_couplings = [J_A] * (n_active - 1) + [J_int] + [J_B] * (n_bath - 1)
    H, hilbert = make_full_tfim_hamiltonian(J_couplings, h=1.0)
    # H, hilbert = make_active_hamiltonian()
    # n_states = hilbert.n_states
    n_states = 2 ** n_active 
    exact_E = get_exact_ground_state_energy(H)

    mixed = MixedAnsatz(bf_input=rng, bf_output=n_states, hilbert=hilbert)

    eta0 = jnp.ones((rng,), jnp.float64)
    sigma0 = jnp.ones((n_active,), jnp.int32)

    global current_eta_local
    current_eta_local = eta0

    params = mixed.init(jax.random.PRNGKey(0), sigma0)["params"]
    params = jax.tree.map(lambda x: x.astype(jnp.float64), params)

    # bf_params = unfreeze(params)["bf"]
    # num_params = sum(x.size for x in jax.tree_util.tree_leaves(bf_params))
    # print(f"[range={rng}] BackflowNN total parameters = {num_params}")

    sampler = nk.sampler.MetropolisLocal(hilbert)
    vstate = MCState(sampler, mixed, n_samples=256)
    vstate.parameters = params

    estimated_energies = [0.0 for _ in range(vmc_iterations)] 

    for η, p in sample_bath(n_bath, J_A, J_B, n_samples_eta):
        current_eta_local = jnp.array(get_local_eta(η, rng), dtype=jnp.float64)
        driver = VMC(hamiltonian=H, optimizer=Sgd(learning_rate),
                     variational_state=vstate, preconditioner=SR(diag_shift=0.1))

        for i in range(vmc_iterations):
            driver.run(n_iter=1, show_progress=False)
            Eη = vstate.expect(H).mean.item()
            estimated_energies[i] += p * Eη    

    rel_errors = [abs(E - exact_E) / abs(exact_E) for E in estimated_energies]
    print(f"[range={rng}] Final relative error = {rel_errors[-1]:.6f}")
    print(f"[range={rng}] Estimated energy = {estimated_energies[-1]} Exact energy = {exact_E}")
    return rel_errors, exact_E


# =====================
# Main
# =====================
if __name__ == "__main__":
    plt.figure(figsize=(10, 5))

    for r in ranges_to_test:
        print(f"\n>>> Backflow range = {r}")
        rel_errors, exact_E = run_backflow(r)
        iterations = np.arange(1, len(rel_errors) + 1)

        plt.plot(iterations, rel_errors, linestyle="--", label=f"Rel Error (range={r})")

    plt.title("Iterations vs. Energy Errors")
    plt.xlabel("Iteration")
    plt.ylabel("Error")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig("abs_rel_error_vs_iterations.png")
    plt.show()

