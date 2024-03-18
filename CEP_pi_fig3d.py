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
h = 6.626 * 10 ** (-34);
ev = 1.602 * 10 ** (-19);
gamma_s = 0.000003 * ev / h;  # atom dissipation rate

# parameter definition
G1  = 0.01 * ev / h / gamma_s;  # coupling strength
J  = 0.0 * ev / h / gamma_s;  # coupling strength
g1  = -0.024 * ev / h / gamma_s;  # coupling strength
w1 = 2.2 * ev / h / gamma_s;  # cavity frequency
wc = 1.2 * ev / h / gamma_s;  # cavity frequency
we = wc - 0.00126 * ev / h / gamma_s;  # atom frequency
kappa_1 = 0.12 * ev / h / gamma_s;       # cavity dissipation rate
kappa =  wc/1000;       # cavity dissipation rate
gamma_s = 1;  # atom dissipation rate
N = 2;              # number of cavity fock states


# operators
a1  = tensor(destroy(N), qeye(N), qeye(N), qeye(2));
c1  = tensor(qeye(N), destroy(N), qeye(N), qeye(2));
c2  = tensor(qeye(N), qeye(N), destroy(N), qeye(2));
sm = tensor(qeye(N), qeye(N), qeye(N), destroy(2));

# Hamiltonian
HL = w1 * a1.dag() * a1 + wc * c1.dag() * c1 + wc * c2.dag() * c2 + we * sm.dag() * sm + G1 * (a1.dag() * sm + a1 * sm.dag()) + g1 * (a1.dag() * c1 + a1 * c1.dag()) + g1 * (a1.dag() * c2 + a1 * c2.dag());
    
p = np.pi
expfac = complex( np.cos(-p), np.sin(-p) )
expfacc = complex( np.cos(p), np.sin(p) )
# dp = c1 + expfac * c2
L11 = kappa * expfacc * ( spre(c1) * spost(c2.dag()) - spre(c2.dag() * c1));
L12 = kappa * expfac * ( spre(c2) * spost(c1.dag()) - spost(c1.dag() * c2));
L2 = gamma_s * ( spre(sm) * spost(sm.dag()) - 0.5 * spost(sm.dag() * sm) - 0.5 * spre(sm.dag() * sm));
L3 = kappa * ( spre(c1) * spost(c1.dag()) - 0.5 * spost(c1.dag() * c1) - 0.5 * spre(c1.dag() * c1));
L4 = kappa * ( spre(c2) * spost(c2.dag()) - 0.5 * spost(c2.dag() * c2) - 0.5 * spre(c2.dag() * c2));
L5 = kappa_1 * ( spre(a1) * spost(a1.dag()) - 0.5 * spost(a1.dag() * a1) - 0.5 * spre(a1.dag() * a1));

L0 = liouvillian(HL);
L = L0 + L11 + L12 + L2 + L3 + L4 + L5; 

c_ops = [];

# # cavity relaxation
# rate = kappa_1;
# if rate > 0.0:
#     c_ops.append(m.sqrt(rate) * a1);
    
# rate = kappa_c;
# if rate > 0.0:
#     c_ops.append(m.sqrt(rate) * c);

# # qubit relaxation
# rate = gamma_s;
# if rate > 0.0:
#     c_ops.append(m.sqrt(rate) * sm);


tlist = np.linspace(0, 0.1, 1001);
# intial state
psi0 = tensor(basis(N,0), basis(N,0), basis(N,0), basis(2,1));    # start with an excited atom

opts = Options()
opts.nsteps = 10**7;

output = mesolve(L, psi0, tlist, c_ops, [a1.dag() * a1, c1.dag() * c1, c2.dag() * c2, sm.dag() * sm], options=opts);

# final plot
n_c = output.expect[0]
n_c1 = output.expect[1]
n_c2 = output.expect[2]
n_a = output.expect[3]

fig, axes = plt.subplots(1, 1, figsize=(8,6))
axes.plot(tlist*G1, n_a, label="Atom excited state")
axes.plot(tlist*G1, n_c1, label="Microcavity 1")
axes.plot(tlist*G1, n_c2, label="Microcavity 2")
axes.plot(tlist*G1, n_c1 + n_c2, label="Microcavity sum")
axes.plot(tlist*G1, n_c, label="Plasmonic cavity")
axes.legend(loc=0)
axes.set_xlabel('Time')
axes.set_ylabel('Occupation probability')
axes.set_title('Vacuum Rabi oscillations')   
 