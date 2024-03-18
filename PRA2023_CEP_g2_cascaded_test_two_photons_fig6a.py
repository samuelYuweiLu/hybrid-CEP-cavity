# matplotlib inline
import matplotlib.pyplot as plt
import numpy as np
import math as m

from qutip import *

# parameter definition
gamma = 0.01        # atom dissipation rate
kappa = 0.1      # cavity dissipation rate: -1.00856
ka = 100
N = 3            # number of cavity fock states
omega = 0.01*gamma
p = np.pi/2
dwa = 1000
# t = -np.pi * 1/4   # simplied conditions
t = -0.257951 * np.pi  # exact conditions
J = 12.5731
gc = 0.0711427 * np.sqrt(2)

dw0 = -J**2*dwa/(dwa**2+ka**2/4)

# operators
c1  = tensor(destroy(N), qeye(N), qeye(N), qeye(2));
c2  = tensor(qeye(N), destroy(N), qeye(N), qeye(2));
a  = tensor(qeye(N), qeye(N), destroy(N), qeye(2));
sm = tensor(qeye(N), qeye(N), qeye(N), destroy(2));


def calulcate_avg_photons(N, dw):
    expfac = complex( np.cos(-p), np.sin(-p) )
    expfacc = complex( np.cos(p), np.sin(p) )
    expthe = complex( np.cos(-t), np.sin(-t) )
    expthec = complex( np.cos(t), np.sin(t) )
    dp1 = ( c2 + c1 ) / np.sqrt(2)
    dp2 = ( c2 - c1 ) / np.sqrt(2)
    L11 = kappa * expfacc * ( spre(c1) * spost(c2.dag()) - spre(c2.dag() * c1));
    L12 = kappa * expfac * ( spre(c2) * spost(c1.dag()) - spost(c1.dag() * c2));
    L2 = gamma * ( spre(sm) * spost(sm.dag()) - 0.5 * spost(sm.dag() * sm) - 0.5 * spre(sm.dag() * sm));
    L3 = kappa * ( spre(c1) * spost(c1.dag()) - 0.5 * spost(c1.dag() * c1) - 0.5 * spre(c1.dag() * c1));
    L4 = kappa * ( spre(c2) * spost(c2.dag()) - 0.5 * spost(c2.dag() * c2) - 0.5 * spre(c2.dag() * c2));
    L5 = ka * ( spre(a) * spost(a.dag()) - 0.5 * spost(a.dag() * a) - 0.5 * spre(a.dag() * a));

    
    H = dw * c1.dag() * c1 + dw * c2.dag() * c2 + (dw + dwa) * a.dag() * a + (dw + dw0) * sm.dag() * sm  \
        + gc * (c1.dag() * sm + c1 * sm.dag()) + J * ( expthe * c1.dag() * a + expthec * c1 * a.dag()) + J * ( expthec * c2.dag() * a + expthe * c2 * a.dag()) \
            + omega * ( sm.dag() + sm )
    L0 = liouvillian(H);
    L = L0 + L11 + L12 + L2 + L3 + L4 + L5    
   
    # Ground state and steady state for the Hamiltonian: H = H0 + g * H1
    rho_ss = steadystate(L)
    # cavity photon number
    n_cavityr = expect(dp1.dag() * dp1, rho_ss)
    n_cavityl = expect(dp2.dag() * dp2, rho_ss)
    n_qe = expect(sm.dag() * sm, rho_ss)
    n_cavity1 = expect(c1.dag() * c1, rho_ss)
    n_cavity2 = expect(c2.dag() * c2, rho_ss)
    # cavity second order coherence function
    g2_cavityr = expect(dp1.dag() * dp1.dag() * dp1 * dp1, rho_ss) / (n_cavityr ** 2)
    g2_cavityl = expect(dp2.dag() * dp2.dag() * dp2 * dp2, rho_ss) / (n_cavityl ** 2)

    return n_cavityr, n_cavityl, n_qe, n_cavity1, n_cavity2, g2_cavityr, g2_cavityl


ncr_vec = []
ncl_vec = []
nqe_vec = []
nc1_vec = []
nc2_vec = []
g2r_vec = []
g2l_vec = []

    
wl_vec = np.linspace(-60*gamma, 60*gamma, 801)

for dw in wl_vec:
    ncr, ncl, nqe, nc1, nc2, g2r, g2l = calulcate_avg_photons(N, dw)
    ncr_vec.append(ncr)
    ncl_vec.append(ncl)
    nqe_vec.append(nqe)
    nc1_vec.append(nc1)
    nc2_vec.append(nc2)
    g2r_vec.append(g2r)
    g2l_vec.append(g2l)



fig, axes = plt.subplots(1, 1, figsize=(6,4))
axes.plot(wl_vec, np.log10(nc1_vec), label="Mode 1")
axes.plot(wl_vec, np.log10(nc2_vec), label="Mode 2")
axes.plot(wl_vec, np.log10(nqe_vec), label="QE")
axes.set_xlabel(r'$\Delta\omega_l (meV)$', fontsize=18);
axes.set_ylabel(r'$n_{ss}$', fontsize=18);
axes.legend(loc=0)


fig, axes = plt.subplots(1, 1, figsize=(6,4))
axes.plot(wl_vec, np.log10(ncr_vec), label="Mode 1")
axes.plot(wl_vec, np.log10(ncl_vec), label="Mode 2")
axes.plot(wl_vec, np.log10(nqe_vec), label="QE")
axes.set_xlabel(r'$\Delta\omega_l (meV)$', fontsize=18);
axes.set_ylabel(r'$n_{ss}$', fontsize=18);
axes.legend(loc=0)


fig, axes = plt.subplots(1, 1, figsize=(6,4))
axes.plot(wl_vec, np.log10(g2r_vec), label="Mode 1")
axes.plot(wl_vec, np.log10(g2l_vec), label="Mode 2")
axes.set_xlabel(r'$\Delta\omega_l (meV)$', fontsize=18);
axes.set_ylabel(r'$g^{(2)}(0)$', fontsize=18);
axes.legend(loc=0)
