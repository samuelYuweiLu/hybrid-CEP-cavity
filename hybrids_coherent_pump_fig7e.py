# -*- coding: utf-8 -*-
"""
Created on Mon Mar 18 00:25:46 2019

@author: luyuwei

"""

# matplotlib inline
import matplotlib.pyplot as plt
import numpy as np
import math as m

from qutip import *

# constant definition
h = 6.626 * 10 ** (-34)
ev = 1.602 * 10 ** (-19)
ys = 0.000003
ym = 0.0000
gamma_s = ym + ys  # atom dissipation rate

# parameter definition
ga = 0.0236  # coupling strength
gc = 0.000171  # coupling strength
# g1 = -0.0029  # coupling strength
kr = 0.013
kn = 0.242
kappa_1 = kn + kr       # cavity dissipation rate
kappa_c = 2.2457/100000       # cavity dissipation rate
kappa_i = kappa_c/100
gamma_z = 0.015
omega = 1*gamma_s
dwa = 2.2543-2.2457
dw = 0
N = 2              # number of cavity fock states


# operators
a = tensor(destroy(N), qeye(N), qeye(N), qeye(2))
c1 = tensor(qeye(N), destroy(N), qeye(N), qeye(2))
c2 = tensor(qeye(N), qeye(N), destroy(N), qeye(2))
sm = tensor(qeye(N), qeye(N), qeye(N), destroy(2)) 
sz = sm.dag()*sm - sm*sm.dag()


def calulcate_avg_photons(g1):
    # Hamiltonian
    H_e = dw * c1.dag() * c1 + dw * c2.dag() * c2 + ( dw + dwa ) * a.dag() * a + dw * sm.dag() * sm + omega * (sm.dag() + sm) + \
        ga * (a.dag() * sm + a * sm.dag()) + gc/np.sqrt(2) * (c1.dag() * sm + c1 * sm.dag()) + gc/np.sqrt(2) * (c2.dag() * sm + c2 * sm.dag()) + \
            (g1/np.sqrt(2) * a.dag() * c1 + g1/np.sqrt(2) * a * c1.dag()) + (g1/np.sqrt(2) * a.dag() * c2 + g1/np.sqrt(2) * a * c2.dag())
            
    L0 = liouvillian(H_e)
    
    p = np.pi * 0
    expfac = complex( np.cos(-p), np.sin(-p) )
    expfacc = complex( np.cos(p), np.sin(p) )
    L11 = kappa_c * expfacc * ( spre(c1) * spost(c2.dag()) - spre(c2.dag() * c1) )
    L12 = kappa_c * expfac * ( spre(c2) * spost(c1.dag()) - spost(c1.dag() * c2) )
    L2 = gamma_s * ( spre(sm) * spost(sm.dag()) - 0.5 * spost(sm.dag() * sm) - 0.5 * spre(sm.dag() * sm) )
    L3 = (kappa_c + kappa_i) * ( spre(c1) * spost(c1.dag()) - 0.5 * spost(c1.dag() * c1) - 0.5 * spre(c1.dag() * c1) )
    L4 = (kappa_c + kappa_i) * ( spre(c2) * spost(c2.dag()) - 0.5 * spost(c2.dag() * c2) - 0.5 * spre(c2.dag() * c2) )
    L5 = kappa_1 * ( spre(a) * spost(a.dag()) - 0.5 * spost(a.dag() * a) - 0.5 * spre(a.dag() * a) )
    L6 = gamma_z * ( spre(sz) * spost(sz.dag()) - 0.5 * spost(sz.dag() * sz) - 0.5 * spre(sz.dag() * sz) )
    
    L = L0 + ( L11 + L12 ) * 1 + L2 + L3 + L4 + L5 + L6 

    # Ground state and steady state for the Hamiltonian: H = H0 + g * H1
    rho_ss = steadystate(L)
    # cavity photon number
    n_qe = expect(sm.dag() * sm, rho_ss)
    n_cavitya = expect(a.dag() * a, rho_ss)
    n_cavity1 = expect(c1.dag() * c1, rho_ss)
    n_cavity2 = expect(c2.dag() * c2, rho_ss)
    n_pe = expect(( np.sqrt(kr)*a.dag() + np.sqrt(ys)*sm.dag() )*( np.sqrt(kr)*a + np.sqrt(ys)*sm ), rho_ss)

    return n_qe, n_cavitya, n_cavity1, n_cavity2, n_pe



nqe_vec = []
nca_vec = []
nc1_vec = []
nc2_vec = []
eta_vec = []

    
g1_vec = np.linspace(-0.04, -0.001, 201)

for g1 in g1_vec:
    nqe, nca, nc1, nc2, n_pe = calulcate_avg_photons(g1)
    nqe_vec.append(nqe)
    nca_vec.append(nca)
    nc1_vec.append(nc1)
    nc2_vec.append(nc2)
    eta_vec.append( (n_pe+kappa_c*(nc1+nc2))/(kn*nca+ym*nqe+n_pe+kappa_c*(nc1+nc2)) )



fig, axes = plt.subplots(1, 1, figsize=(6,4))
axes.plot(g1_vec*1000, np.log10(nca_vec), label="Plasmon")
axes.plot(g1_vec*1000, np.log10(nc1_vec), label="CCW")
axes.plot(g1_vec*1000, np.log10(nc2_vec), label="CW")
axes.plot(g1_vec*1000, np.log10(nqe_vec), label="QE")
axes.set_xlabel(r'$\Delta\omega_L / \kappa_c$', fontsize=18);
axes.set_ylabel(r'$Occupation$', fontsize=18);
axes.legend(loc=0)


fig, axes = plt.subplots(1, 1, figsize=(6,4))
axes.plot(g1_vec*1000, eta_vec, label="Quantum yield")
axes.set_xlabel(r'g_1', fontsize=18);
axes.set_ylabel(r'Quantum yield $\eta$', fontsize=18);
# axes.legend(loc=0)

