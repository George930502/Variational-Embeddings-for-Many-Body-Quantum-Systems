import os
import numpy as np
from collections import Counter
from itertools import product
import random

from pyscf import gto, scf, lo, fci

import netket as nk
import netket.experimental as nkx

import flax.linen as nn
import jax.numpy as jnp
import jax

mol = gto.M(atom='H 0 0 0; H 0 0 0.75; H 0 0 1.5; H 0 0 2.25', basis='sto-3g')   # H4 chain for example case
mol.build()
mf = scf.RHF(mol).run()
n_orb = mf.mo_coeff.shape[1]

cisolver = fci.FCI(mol, mf.mo_coeff)
fci_energy = cisolver.kernel()[0]

n_active = 2
n_bath = n_orb - n_active
mo_coeff = lo.Boys(mol).kernel(mf.mo_coeff)   # Localized molecular orbitals
mo_active = mo_coeff[:, :n_active]    # How molecular orbitals are linearly combined from atomic orbitals
mo_bath = mo_coeff[:, n_active:]

H = nkx.operator.from_pyscf_molecule(mol, mo_coeff=mo_active)  # The second-quantized Hamiltonian acting on the active space
hi_active = H.hilbert

class ConditionalMLP(nn.Module):
    '''
    Return: log psi(sigma | eta)
    '''
    n_active: int
    n_bath: int
    hidden_dim: int = 64  # tunable

    @nn.compact
    def __call__(self, sigma_eta):
        x = sigma_eta.astype(jnp.float32)  
        x = nn.Dense(self.hidden_dim)(x)
        x = nn.relu(x)                     
        x = nn.Dense(self.hidden_dim)(x)
        x = nn.relu(x)
        x = nn.Dense(1)(x)

        # log-amplitude
        return jnp.squeeze(x, axis=-1)
    
class ConditionedState(nk.vqs.MCState):
    def __init__(self, sampler, model, params, eta_array, n_samples=1024, debug=False):
        self.debug = debug

        def apply_fun(params, sigma, **kwargs):
            eta_broadcasted = jnp.broadcast_to(eta_array, sigma.shape[:-1] + eta_array.shape)
            sigma_eta = jnp.concatenate([sigma, eta_broadcasted], axis=-1)

            if debug:
                print("sigma shape:", sigma.shape)
                print("sigma_eta shape:", sigma_eta.shape)

            return model.apply(params, sigma_eta, **kwargs)
        
        super().__init__(
            sampler=sampler,
            apply_fun=apply_fun,
            variables=params,
            n_samples=n_samples,
        )

def sample_bath_distribution(n_bath, n_samples):
    configs = list(product([0, 1], repeat=n_bath))
    sampled = random.choices(configs, k=n_samples)
    counter = Counter(sampled)
    result = [(np.array(eta), count / n_samples) for eta, count in counter.items()]  # (eta, p_eta)
    return result

eta_distribution = sample_bath_distribution(n_bath, 500)

E_total = 0.0
E_eta_list = [] # record each eta expectation

for eta_array, p_eta in eta_distribution:
    sampler = nk.sampler.MetropolisLocal(hi_active)

    n_active = hi_active.size  
    n_bath = eta_array.shape[0]

    # print(n_active, n_bath)
    
    flax_model = ConditionalMLP(n_active=n_active, n_bath=n_bath)

    # Mpdel Initialization
    dummy_sigma = jnp.zeros((1, n_active))
    dummy_eta = jnp.reshape(eta_array, (1, n_bath)) 
    dummy_input = jnp.concatenate([dummy_sigma, dummy_eta], axis=-1)
    params = flax_model.init(jax.random.PRNGKey(0), dummy_input)

    vstate = ConditionedState(sampler, flax_model, params, eta_array)

    # VMC optimization
    opt = nk.optimizer.Sgd(learning_rate=0.01)
    sr = nk.optimizer.SR(diag_shift=0.1)
    driver = nk.driver.VMC(H, optimizer=opt, variational_state=vstate, preconditioner=sr)
    driver.run(n_iter=500, show_progress=False)  

    # Estimate E_η
    energy_stats = vstate.expect(H)
    E_eta = energy_stats.mean.item()  
    weighted_energy = p_eta * E_eta

    print(f"E_eta: {E_eta:.6f}, p_eta: {p_eta:.4f}, weighted: {weighted_energy:.6f}")
    E_eta_list.append((eta_array, p_eta, E_eta))

    E_total += weighted_energy

relative_error = abs(E_total - fci_energy) / abs(fci_energy)

print("FCI Energy:", fci_energy)
print("VMC Estimated Energy (E_total):", E_total)
print("Relative Error:", relative_error * 100, "%")