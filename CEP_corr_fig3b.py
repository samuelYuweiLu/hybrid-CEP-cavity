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
gamma_0 = 0.003
gamma_m = 0.0
gamma_s = gamma_0 + gamma_m  # atom dissipation rate

# parameter definition
G1  = 10  # coupling strength
J  = 0  # coupling strength
g1  = -24  # coupling strength
w1 = 2200  # cavity frequency
wc = 1200  # cavity frequency
we = 1199.1  # atom frequency
kappa_1 = 120       # cavity dissipation rate
kappa = wc/10**3       # cavity dissipation rate
# gamma_s = 0;  # atom dissipation rate
N = 2              # number of cavity fock states


# operators
a1 = tensor(destroy(N), qeye(N), qeye(N), qeye(2))
c1 = tensor(qeye(N), destroy(N), qeye(N), qeye(2))
c2 = tensor(qeye(N), qeye(N), destroy(N), qeye(2))
sm = tensor(qeye(N), qeye(N), qeye(N), destroy(2))


# Hamiltonian
HL = w1 * a1.dag() * a1 + wc * c1.dag() * c1 + wc * c2.dag() * c2 + we * sm.dag() * sm + J * (c1.dag() * sm + c1 * sm.dag()) + J * (c2.dag() * sm + c2 * sm.dag()) + \
    G1 * (a1.dag() * sm + a1 * sm.dag()) + g1 * (a1.dag() * c1 + a1 * c1.dag()) + g1 * (a1.dag() * c2 + a1 * c2.dag())
    
p = np.pi*0.75
expfac = complex( np.cos(-p), np.sin(-p) )
expfacc = complex( np.cos(p), np.sin(p) )
# dp = c1 + expfac * c2
L11 = kappa * expfacc * ( spre(c1) * spost(c2.dag()) - spre(c2.dag() * c1))
L12 = kappa * expfac * ( spre(c2) * spost(c1.dag()) - spost(c1.dag() * c2))
L2 = gamma_s * ( spre(sm) * spost(sm.dag()) - 0.5 * spost(sm.dag() * sm) - 0.5 * spre(sm.dag() * sm))
L3 = kappa * ( spre(c1) * spost(c1.dag()) - 0.5 * spost(c1.dag() * c1) - 0.5 * spre(c1.dag() * c1))
L4 = kappa * ( spre(c2) * spost(c2.dag()) - 0.5 * spost(c2.dag() * c2) - 0.5 * spre(c2.dag() * c2))
L5 = kappa_1 * ( spre(a1) * spost(a1.dag()) - 0.5 * spost(a1.dag() * a1) - 0.5 * spre(a1.dag() * a1))

L0 = liouvillian(HL)
L = L0 + L11 + L12 + L2 + L3 + L4 + L5


c_ops = []

# intial state
psi0 = tensor(basis(N,0), basis(N,0), basis(N,0), basis(2,1))    # start with an excited atom

# wl_vec = np.linspace(-3, 1, 201)


        
tlist = np.linspace(0, 500, 5000)
corr = correlation_2op_1t(L, psi0, tlist, c_ops, sm.dag(), sm)
wlist1, spec1 = spectrum_correlation_fft(tlist, corr)
# calculate the power spectrum using spectrum, which internally uses essolve
# to solve for the dynamics (by default)
# wlist2 = np.linspace(1197, 1201, 200)
# spec2 = spectrum(L, wlist2, c_ops, sm.dag(), sm)
# plot the spectra
fig, ax = plt.subplots(1, 1)
ax.plot(wlist1, spec1, 'b', lw=2, label='eseries method')
# ax.plot(wlist2 / (2 * np.pi), spec2, 'r--', lw=2, label='me+fft method')
ax.legend()
ax.set_xlabel('Frequency')
ax.set_ylabel('Power spectrum')
ax.set_title('Vacuum Rabi splitting')
ax.set_xlim(4.5, 6.5)
            




  
 