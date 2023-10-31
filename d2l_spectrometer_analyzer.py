# -*- coding: utf-8 -*-
"""
Created on Mon Oct 30 11:14:03 2023

@author: lisaf
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

def gaussian(x, N, mu, sigma):
    return N*np.exp(-0.5*((x-mu)/sigma)**2)

chanel, counts = np.loadtxt('calib_cesium_137', unpack = True, skiprows=10, delimiter =';', usecols=(0,1))

fig = plt.figure()
ax = fig.add_subplot()
ax.plot(chanel, counts, color = 'gray')
ax.set_xlabel('Channel')
ax.set_ylabel('Counts')
ax.grid()

#calibration
chanel_cal = [55, 139, 1065]
energy_cal = [32, 81, 662]

coeff = np.polyfit(chanel_cal, energy_cal, deg = 1)
energies = chanel*coeff[0] + coeff[1]

fig = plt.figure()
ax = fig.add_subplot()
ax.plot(energies, counts, color = 'gray')
ax.set_xlabel('Energy [keV]')
ax.set_ylabel('Counts')
ax.grid()

#fit peak
energy_min = 650
energy_max = 680
mask_fit = np.logical_and(energies>=energy_min, energies<=energy_max)
energies_to_fit = energies[mask_fit]
counts_to_fit = counts[mask_fit]
par, cov = curve_fit(gaussian, xdata = energies_to_fit, ydata = counts_to_fit, bounds = ((0,0,0),(np.max(counts),2000,100)))
errs = np.sqrt(np.diag(cov))
y_fit = gaussian(energies_to_fit, *par)

fig = plt.figure()
ax = fig.add_subplot()
ax.plot(energies, counts, color = 'gray', label = 'Data')
ax.plot(energies_to_fit, y_fit, color = 'red', label = 'Fit')
ax.set_xlabel('Energy [keV]')
ax.set_ylabel('Counts')
ax.set_xlim(600,700)
ax.set_ylim(0,200)
ax.legend()
ax.grid()

print('PEAK ENERGY: {0:.2f}+/-{1:.2f} keV'.format(par[1], errs[1]))
print('FWHM: {0:.2f}+/-{1:.2f} keV'.format(par[2]*2.355, errs[2]*2.355))
print('Energy Res: {0:.2f}%'.format(par[2]*2.355/par[1]*100))
