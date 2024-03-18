# -*- coding: utf-8 -*-
"""
Created on Tue Dec 15 13:08:34 2020

@author: luyuwei
"""

#%matplotlib inline
#%config InlineBackend.figure_format = 'retina'

import matplotlib.pyplot as plt
import numpy as np

from scipy.interpolate import interp2d

# from Math import *
from qutip import *

# shared parameters
y0 = 0.00001
wc = 1.6/y0;
w1 = wc;
gamma = y0 / y0                        # decay rate
gammaz = 0.02 / y0
kappa = 0.002 / y0
kappa1 = 0.1 / y0
gc = 0.000 / y0
ga = 0.031 / y0
g1 = - 0.011 / y0
dw = 0
p = np.pi*0
N = 2

#ga = gc*kappa1/2/(0.02/y0);
#dw0 = (-((np.abs(g1)*gc)/ga) + (ga*gc)/np.abs(g1))*np.cos(p1);
dw0 = 0.0
wa = wc + dw0;


tmax = 0.15
tlist = np.linspace(0, tmax, 1500)
taulist = tlist[0:int(len(tlist)/2-1)]


# intial state
psi0 = tensor(basis(N,0), basis(N,0), basis(N,0), basis(2,1));    # start with an excited atom

# operators
a  = tensor(destroy(N), qeye(N), qeye(N), qeye(2))
c1  = tensor(qeye(N), destroy(N), qeye(N), qeye(2))
c2  = tensor(qeye(N), qeye(N), destroy(N), qeye(2))
sm = tensor(qeye(N), qeye(N), qeye(N), destroy(2))             # atomic lowering operator

# Hamiltonian
H_e = ( dw + 0 ) * c1.dag() * c1 + ( dw + 0 ) * c2.dag() * c2 + dw * a.dag() * a + ( dw + dw0 ) * sm.dag() * sm + \
    ga * (a.dag() * sm + a * sm.dag()) + gc/np.sqrt(2) * (c1.dag() * sm + c1 * sm.dag()) + gc/np.sqrt(2) * (c2.dag() * sm + c2 * sm.dag()) + \
        (g1/np.sqrt(2) * a.dag() * c1 + g1/np.sqrt(2) * a * c1.dag()) + (g1/np.sqrt(2) * a.dag() * c2 + g1/np.sqrt(2) * a * c2.dag())
        
L0 = liouvillian(H_e)

# expfac = complex( np.cos(-p), np.sin(-p) )
# expfacc = complex( np.cos(p), np.sin(p) )
# L1 = kappa * expfacc * ( spre(c1) * spost(c2.dag()) - spre(c2.dag() * c1))
# L2 = kappa * expfac * ( spre(c2) * spost(c1.dag()) - spost(c1.dag() * c2))

L = L0

# collapse operator that describes dissipation
c_ops = [np.sqrt(gamma) * sm, np.sqrt(gammaz) * sm.dag()*sm, np.sqrt(kappa) * c1, np.sqrt(kappa) * c2, np.sqrt(kappa1) * a]  # represents spontaneous emission
# number operator
n = [sm.dag()*sm, c1.dag()*c1, c2.dag()*c2] 


options = Options()
options.nsteps = 10000
options.max_step = (tlist[1]-tlist[0])/2

ms = mesolve(L, psi0, tlist, c_ops, n, options=options)
n_e = ms.expect[0]
n_c1 = ms.expect[1]
n_c2 = ms.expect[2]
# n_ct = ms.expect[3]

fig, ax = plt.subplots(figsize=(8,6))
ax.plot(tlist, n_e/100, 'r', label="emitter")
ax.plot(tlist, n_c1, 'b', label="cavity 1")
ax.plot(tlist, n_c2, 'c', label="cavity 2")
# ax.plot(tlist, n_ct, 'k', label="WG output")
# ax.plot(tlist, pulse_shape_e/10000, '-k', label="excitation: exponential wavepacket")
ax.legend()
ax.set_xlim(0, tmax)
# ax.set_ylim(0, 1)
ax.set_xlabel('Time, $t$ [$1/\gamma$]')
ax.set_ylabel('Emission flux [$\gamma$]')
ax.set_title('TLS/Cavity emission shapes');


# specify relevant operators to calculate the correlation
# <A(t)B(t+tau)C(t)>
a_op = sm.dag()
b_op = sm.dag() * sm
c_op = sm

# calculate two-time correlations
# G2_t_tau_e0 = correlation_2op_2t(L, psi0, tlist[0:int(len(tlist)/2-1)], taulist, c_ops, a_op, c_op, options=options)
G2_t_tau_c10 = correlation_2op_2t(L, psi0, tlist[0:int(len(tlist)/2-1)], taulist, c_ops, c1.dag(), c1, options=options)
G2_t_tau_c20 = correlation_2op_2t(L, psi0, tlist[0:int(len(tlist)/2-1)], taulist, c_ops, c2.dag(), c2, options=options)
# G2_t_tau_ct0 = correlation_2op_2t(L, psi0, tlist[0:int(len(tlist)/2-1)], taulist, c_ops, c1.dag()*expfac + c2.dag(), c1*expfacc + c2, options=options)
# G2_t_tau_e2 = np.abs(G2_t_tau_e0)**2
G2_t_tau_c12 = np.abs(G2_t_tau_c10)**2
G2_t_tau_c22 = np.abs(G2_t_tau_c20)**2
# G2_t_tau_ct2 = np.abs(G2_t_tau_ct0)**2

# n_t_tau_e = []
n_t_tau_c1 = []
n_t_tau_c2 = []
# n_t_tau_ct = []
for i in range(0,len(tlist),1):
    # n_t_tau_e.append(np.sum(n_e[i:])*(tlist[1]-tlist[0]))
    n_t_tau_c1.append(np.sum(n_c1[i:])*(tlist[1]-tlist[0]))
    n_t_tau_c2.append(np.sum(n_c2[i:])*(tlist[1]-tlist[0]))
    # n_t_tau_ct.append(np.sum(n_ct[i:])*(tlist[1]-tlist[0]))

# n_t_e = np.multiply(n_e, n_t_tau_e)
n_t_c1 = np.multiply(n_c1, n_t_tau_c1)
n_t_c2 = np.multiply(n_c2, n_t_tau_c2)
# n_t_ct = np.multiply(n_ct, n_t_tau_ct)

# Id_e = np.sum(n_t_e)*(tlist[1]-tlist[0])
Id_c1 = np.sum(n_t_c1)*(tlist[1]-tlist[0])
Id_c2 = np.sum(n_t_c2)*(tlist[1]-tlist[0])
# Id_ct = np.sum(n_t_ct)*(tlist[1]-tlist[0])

# I_e = (G2_t_tau_e2.sum(0)).sum(0)*(tlist[1]-tlist[0])**2/Id_e
I_c1 = (G2_t_tau_c12.sum(0)).sum(0)*(tlist[1]-tlist[0])**2/Id_c1
I_c2 = (G2_t_tau_c22.sum(0)).sum(0)*(tlist[1]-tlist[0])**2/Id_c2
# I_ct = (G2_t_tau_ct2.sum(0)).sum(0)*(tlist[1]-tlist[0])**2/Id_ct

fig = plt.figure(figsize=(8,5))

# ax_e = fig.add_subplot(121)
# p_e = ax_e.pcolor(tlist[0:int(len(tlist)/2-1)]*gamma, taulist*gamma, np.abs(G2_t_tau_e0).transpose())
# ax_e.set_xlim(0, tmax/2)
# ax_e.set_ylim(0, tmax/2)
# ax_e.set_xlabel('Time, $t$ [$1/\gamma$]')
# ax_e.set_ylabel('Delay, $\\tau$ [$1/\gamma$]')
# ax_e.set_title('$G^{(2)}(t,\\tau)$ for emitter');

ax_G = fig.add_subplot(121)
p_G = ax_G.pcolor(tlist[0:int(len(tlist)/2-1)]*gamma, taulist*gamma, np.abs(G2_t_tau_c10).transpose())
ax_G.set_xlim(0, tmax/2)
ax_G.set_ylim(0, tmax/2)
ax_G.set_xlabel('Time, $t$ [$1/\gamma$]')
ax_G.set_ylabel('Delay, $\\tau$ [$1/\gamma$]')
ax_G.set_title('$G^{(2)}(t,\\tau)$ for cavity 1');

ax_G2 = fig.add_subplot(122)
p_G2 = ax_G2.pcolor(tlist[0:int(len(tlist)/2-1)]*gamma, taulist*gamma, np.abs(G2_t_tau_c20).transpose())
ax_G2.set_xlim(0, tmax/2)
ax_G2.set_ylim(0, tmax/2)
ax_G2.set_xlabel('Time, $t$ [$1/\gamma$]')
ax_G2.set_ylabel('Delay, $\\tau$ [$1/\gamma$]')
ax_G2.set_title('$G^{(2)}(t,\\tau)$ for cavity 2');



fig, ax = plt.subplots(figsize=(8,5))
# ax.plot(taulist, abs(G2_t_tau_e0[:,0])/20, 'r', label="emitter")
ax.plot(taulist, abs(G2_t_tau_c10[:,0]), 'b', label="cavity, $t$")
ax.plot(taulist, abs(G2_t_tau_c10[70,:]), '--b', label="cavity, $\\tau$")
ax.legend()
ax.set_xlim(0, tmax/2)
# ax.set_ylim(0, 0.07)
ax.set_xlabel('Time delay, $\\tau$ [$1/\gamma$]')
ax.set_ylabel('$G^{(2)}(\\tau)$ [$\gamma^2$]')
ax.set_title('Integrated second-order coherence');

fig, ax = plt.subplots(figsize=(8,5))
# ax.plot(taulist, abs(G2_t_tau_e0[:,0])/20, 'r', label="emitter")
ax.plot(taulist, abs(G2_t_tau_c20[:,0]), 'b', label="cavity, $t$")
ax.plot(taulist, abs(G2_t_tau_c20[70,:]), '--b', label="cavity, $\\tau$")
ax.legend()
ax.set_xlim(0, tmax/2)
# ax.set_ylim(0, 0.07)
ax.set_xlabel('Time delay, $\\tau$ [$1/\gamma$]')
ax.set_ylabel('$G^{(2)}(\\tau)$ [$\gamma^2$]')
ax.set_title('Integrated second-order coherence');


# display('emitter, I = ' + str(round(I_e, 12)))
display('cavity 1, I = ' + str(round(I_c1, 12)))
display('cavity 2, I = ' + str(round(I_c2, 12)))
# display('WG output, I = ' + str(round(I_ct, 12)))

print(abs(np.trapz(n_e[0:int(len(tlist)/2-1)], taulist*gamma))*gamma)
print(abs(np.trapz(n_c1[0:int(len(tlist)/2-1)], taulist*gamma))*kappa)
print(abs(np.trapz(n_c2[0:int(len(tlist)/2-1)], taulist*gamma))*kappa)
# print(abs(np.trapz(n_ct[0:int(len(tlist)/2-1)], taulist*gamma))*kappa)
display('cavity total = ' + str(round(abs(np.trapz(n_c1[0:int(len(tlist)/2-1)]+n_c2[0:int(len(tlist)/2-1)], taulist*gamma))*kappa, 12)))
# display('WG total = ' + str(round(abs(np.trapz(n_ct[0:int(len(tlist)/2-1)], taulist*gamma))*kappa, 12)))



