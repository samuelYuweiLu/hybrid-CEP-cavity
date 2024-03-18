# -*- coding: utf-8 -*-
"""
Created on Tue Dec 15 13:08:34 2020

@author: luyuwei
"""

#%matplotlib inline
#%config InlineBackend.figure_format = 'retina'

import matplotlib.pyplot as plt
import numpy as np
import scipy.io

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
gc = 0.00 / y0
ga = 0.031 / y0
g1 = - 0.011 / y0
dw = 0
p_list = np.linspace( 0, 2*np.pi, 101 )
N = 2

#ga = gc*kappa1/2/(0.02/y0);
#dw0 = (-((np.abs(g1)*gc)/ga) + (ga*gc)/np.abs(g1))*np.cos(p1);
dw0 = 0;
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


def calulcate_cavity_I(p):
    # Hamiltonian
    H_e = ( dw + dw0 ) * c1.dag() * c1 + ( dw + dw0 ) * c2.dag() * c2 + dw * a.dag() * a + dw * sm.dag() * sm + \
        ga * (a.dag() * sm + a * sm.dag()) + gc/np.sqrt(2) * (c1.dag() * sm + c1 * sm.dag()) + gc/np.sqrt(2) * (c2.dag() * sm + c2 * sm.dag()) + \
            (g1/np.sqrt(2) * a.dag() * c1 + g1/np.sqrt(2) * a * c1.dag()) + (g1/np.sqrt(2) * a.dag() * c2 + g1/np.sqrt(2) * a * c2.dag())
            
    L0 = liouvillian(H_e)
    
    expfac = complex( np.cos(-p), np.sin(-p) )
    expfacc = complex( np.cos(p), np.sin(p) )
    L1 = kappa * expfacc * ( spre(c1) * spost(c2.dag()) - spre(c2.dag() * c1))
    L2 = kappa * expfac * ( spre(c2) * spost(c1.dag()) - spost(c1.dag() * c2))
    
    L = L0 + L1 + L2
    
    # collapse operator that describes dissipation
    c_ops = [np.sqrt(gamma) * sm, np.sqrt(gammaz) * sm.dag()*sm, np.sqrt(kappa) * c1, np.sqrt(kappa) * c2, np.sqrt(kappa1) * a]  # represents spontaneous emission
    # number operator
    n = [sm.dag()*sm, c1.dag()*c1, c2.dag()*c2, (c1.dag()*expfac+c2.dag())*(c1*expfacc+c2)] 
    
    
    options = Options()
    options.nsteps = 10000
    options.max_step = (tlist[1]-tlist[0])/2
    
    ms = mesolve(L, psi0, tlist, c_ops, n, options=options)
    n_e = ms.expect[0]
    n_c1 = ms.expect[1]
    n_c2 = ms.expect[2]
    n_ct = ms.expect[3]
    
    beta_c1 = abs(np.trapz(n_c1[0:int(len(tlist)/2-1)], taulist*gamma))*kappa
    beta_c2 = abs(np.trapz(n_c2[0:int(len(tlist)/2-1)], taulist*gamma))*kappa
    beta_ct = abs(np.trapz(n_ct[0:int(len(tlist)/2-1)], taulist*gamma))*kappa


    # specify relevant operators to calculate the correlation
    # <A(t)B(t+tau)C(t)>
    a_op = sm.dag()
    b_op = sm.dag() * sm
    c_op = sm
    
    # calculate two-time correlations
    # G2_t_tau_e0 = correlation_2op_2t(L, psi0, tlist[0:int(len(tlist)/2-1)], taulist, c_ops, a_op, c_op, options=options)
    G2_t_tau_c10 = correlation_2op_2t(L, psi0, tlist[0:int(len(tlist)/2-1)], taulist, c_ops, c1.dag(), c1, options=options)
    G2_t_tau_c20 = correlation_2op_2t(L, psi0, tlist[0:int(len(tlist)/2-1)], taulist, c_ops, c2.dag(), c2, options=options)
    G2_t_tau_ct0 = correlation_2op_2t(L, psi0, tlist[0:int(len(tlist)/2-1)], taulist, c_ops, c1.dag()*expfac + c2.dag(), c1*expfacc + c2, options=options)
    # G2_t_tau_e2 = np.abs(G2_t_tau_e0)**2
    G2_t_tau_c12 = np.abs(G2_t_tau_c10)**2
    G2_t_tau_c22 = np.abs(G2_t_tau_c20)**2
    G2_t_tau_ct2 = np.abs(G2_t_tau_ct0)**2
    
    # n_t_tau_e = []
    n_t_tau_c1 = []
    n_t_tau_c2 = []
    n_t_tau_ct = []
    for i in range(0,len(tlist),1):
        # n_t_tau_e.append(np.sum(n_e[i:])*(tlist[1]-tlist[0]))
        n_t_tau_c1.append(np.sum(n_c1[i:])*(tlist[1]-tlist[0]))
        n_t_tau_c2.append(np.sum(n_c2[i:])*(tlist[1]-tlist[0]))
        n_t_tau_ct.append(np.sum(n_ct[i:])*(tlist[1]-tlist[0]))
    
    # n_t_e = np.multiply(n_e, n_t_tau_e)
    n_t_c1 = np.multiply(n_c1, n_t_tau_c1)
    n_t_c2 = np.multiply(n_c2, n_t_tau_c2)
    n_t_ct = np.multiply(n_ct, n_t_tau_ct)
    
    # Id_e = np.sum(n_t_e)*(tlist[1]-tlist[0])
    Id_c1 = np.sum(n_t_c1)*(tlist[1]-tlist[0])
    Id_c2 = np.sum(n_t_c2)*(tlist[1]-tlist[0])
    Id_ct = np.sum(n_t_ct)*(tlist[1]-tlist[0])
    
    # I_e = (G2_t_tau_e2.sum(0)).sum(0)*(tlist[1]-tlist[0])**2/Id_e
    I_c1 = (G2_t_tau_c12.sum(0)).sum(0)*(tlist[1]-tlist[0])**2/Id_c1
    I_c2 = (G2_t_tau_c22.sum(0)).sum(0)*(tlist[1]-tlist[0])**2/Id_c2
    I_ct = (G2_t_tau_ct2.sum(0)).sum(0)*(tlist[1]-tlist[0])**2/Id_ct


    return n_e, n_c1, n_c2, n_ct, I_c1, I_c2, I_ct, beta_c1, beta_c2, beta_ct


n_e_list = np.zeros((len(p_list), len(tlist)), dtype = float )
n_c1_list = np.zeros((len(p_list), len(tlist)), dtype = float)
n_c2_list = np.zeros((len(p_list), len(tlist)), dtype = float)
n_ct_list = np.zeros((len(p_list), len(tlist)), dtype = float)
I_c1_list = np.zeros((len(p_list)), dtype = float)
I_c2_list = np.zeros((len(p_list)), dtype = float)
I_ct_list = np.zeros((len(p_list)), dtype = float)
eta_c1_list = np.zeros((len(p_list)), dtype = float)
eta_c2_list = np.zeros((len(p_list)), dtype = float)
eta_ct_list = np.zeros((len(p_list)), dtype = float)


for i in range(0,len(p_list),1):
    n_e_m, n_c1_m, n_c2_m, n_ct_m, I_c1_m, I_c2_m, I_ct_m, eta_c1_m, eta_c2_m, eta_ct_m = calulcate_cavity_I(p_list[i])

    n_e_list[i,:] = n_e_m
    n_c1_list[i,:] = n_c1_m
    n_c2_list[i,:] = n_c2_m
    n_ct_list[i,:] = n_ct_m
    I_c1_list[i] = I_c1_m
    I_c2_list[i] = I_c2_m
    I_ct_list[i] = I_ct_m
    eta_c1_list[i] = eta_c1_m
    eta_c2_list[i] = eta_c2_m
    eta_ct_list[i] = eta_ct_m

    print('CCW Indistinguishability, I = ' + str(round(I_c1_m, 12)))
    print('CW Indistinguishability, I = ' + str(round(I_c2_m, 12)))
    print('WG Indistinguishability, I = ' + str(round(I_ct_m, 12)))
    print('CCW efficiency = ' + str(round(eta_c1_m, 12)))
    print('CW efficiency = ' + str(round(eta_c2_m, 12)))
    print('WG efficiency = ' + str(round(eta_ct_m, 12))) 
    print('No. ' + str(round(i+1, 12)) + ' is finished. Total:' + str(round(len(p_list)))) 
    print('__________________________')
    print(' ')


scipy.io.savemat('I_p_scan_opt.mat', mdict={'n_e_list': n_e_list, 'n_c1_list': n_c1_list, 'n_c2_list': n_c2_list, 'n_ct_list': n_ct_list, \
                                        'I_c1_list': I_c1_list, 'I_c2_list': I_c2_list, 'I_ct_list': I_ct_list, 'eta_c1_list': eta_c1_list, \
                                            'eta_c2_list': eta_c2_list, 'eta_ct_list': eta_ct_list, 'p_list': p_list})



# fig, axes = plt.subplots(1, 1, sharex=True, figsize=(6,4))
# axes.plot(p_list/np.pi, I_c1_list,  'b', label=r'Occupation number')
# axes.set_xlabel(r'$\phi / \pi$')

# fig, axes = plt.subplots(1, 1, sharex=True, figsize=(6,4))
# axes.plot(p_list/np.pi, I_c2_list,  'b', label=r'Occupation number')
# axes.set_xlabel(r'$\phi / \pi$')

fig, axes = plt.subplots(1, 1, sharex=True, figsize=(6,4))
axes.plot(p_list/np.pi, I_ct_list,  'b', label=r'Occupation number')
axes.set_xlabel(r'$\phi / \pi$')

# fig, axes = plt.subplots(1, 1, sharex=True, figsize=(6,4))
# axes.plot(p_list/np.pi, eta_c1_list,  'b', label=r'Efficiency $\eta$')
# axes.set_xlabel(r'$\phi / \pi$')

# fig, axes = plt.subplots(1, 1, sharex=True, figsize=(6,4))
# axes.plot(p_list/np.pi, eta_c2_list,  'b', label=r'Efficiency $\eta$')
# axes.set_xlabel(r'$\phi / \pi$')

fig, axes = plt.subplots(1, 1, sharex=True, figsize=(6,4))
axes.plot(p_list/np.pi, eta_ct_list,  'b', label=r'Efficiency $\eta$')
axes.set_xlabel(r'$\phi / \pi$')






