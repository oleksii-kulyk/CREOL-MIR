import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as sig

import dsatools
from dsatools import operators
import dsatools.utilits as ut
from dsatools import decomposition



#-------------DEBUG FLAGS
# SET TO TRUE TO SEE THE PLOTS INTERACTIVELY
SHOWPLT = True
# SET TO TRUE TO SAVE THE PLOTS
SAVEFIG = False
# PLOT SAVE FORMAT png - for raster graphics; svg - for vector graphics
FORMATPLT = 'png'
# PLOT SAVE DPI
DPI = 1200
# PLOT LINE WIDTH
LINEWIDTH = 0.5

#-------------CONSTANTS
n = 207_964
FrepLO = 115_222_004.733 #Hz
c = 29979245800 #cm/s



#------------------------------------------------------
# This Section Preforms Normalization, Outlier Disrading, and Baseline Detection

# generate x-axis for breath spectrum
x_axis = np.array([ i*138.88888888888889 for i in range(450_000) ])
# shift to proper frequency range; relation in Formula.txt file in COPD_Test Data
x_axis = ( 95_000_000 + (n+1) * (4 * FrepLO - x_axis) ) / c
# formula above reverses the array; reverse it back to monotinically increase
x_axis = np.flip(x_axis)

# import breath spectra as y-axis
breath = np.loadtxt("Demo Data/2022_0719_1851_x8-207964_tau1800us_Triangle_zpC1_Frep115222004p733_empty_20954742avgs_SPCT.txt", skiprows=3)
# normalize the breath spectra
breath = (breath - np.min(breath)) / (np.max(breath) - np.min(breath))
# reverse the y-axis respectively with the x-axis
breath = np.flip(breath)

# inspecting preliminary data
plt.plot(x_axis, breath, alpha=0.3, label="Raw Data", linewidth=LINEWIDTH)


# Use a HVD with a LO-pass of emipirical 150Hz to obtain a baseline
baseline = decomposition.hvd(breath, order=1, fpar=150)
# Plot baseline to visually confirm proper LO-pass freq selection
plt.plot(x_axis, baseline[0], label="Inferred Baseline", linewidth=LINEWIDTH)

plt.legend()
if SAVEFIG: plt.savefig(f"01 Normalized Data With Baseline.{FORMATPLT}", format=FORMATPLT, dpi=DPI)
if SHOWPLT: plt.show()
plt.clf()


# Subtract baseline from y-axis
breath -= baseline[0]
# Plot after subtracting the baseline
plt.plot(x_axis, breath, label="Subtracted Baseline", linewidth=LINEWIDTH)

# After visual inspection, delete outliers from start and from end of the array
x_axis_trim = np.delete(x_axis, np.s_[0:1500:])
x_axis_trim = np.delete(x_axis_trim, np.s_[-6000::])
breath_trim = np.delete(breath, np.s_[0:1500:])
breath_trim = np.delete(breath_trim, np.s_[-6000::])

# check how much was trimmed
plt.plot(x_axis_trim, breath_trim - 0.05, label="Trimmed Data No Baseline", linewidth=LINEWIDTH)

plt.legend()
if SAVEFIG: plt.savefig(f"02 Trimmed Normalized Data No Baseline.{FORMATPLT}", format=FORMATPLT, dpi=DPI)
if SHOWPLT: plt.show()
plt.clf()



#------------------------------------------------------
# This Section Preforms Measurement Denoizing

#TODO Impelement Raman Spectroscopy Methods for Denoizing
#ut.probe(psds.sum(axis=0))

#TODO Experiment with Bilateral Filter

# Preform Hilbert Vibration Decomposition with LO-pass filter of 50K Hz
psds = decomposition.hvd(breath_trim, order=6, fpar=5000)
#print(f"Hilbert Vibration Decomposition {psds}")


# plot the decomposition
for i, order in enumerate(psds):
  plt.plot(x_axis_trim, order - (i * 0.005), label=f"HVD Component Order {i+1}", linewidth=LINEWIDTH)

plt.plot(x_axis_trim, psds.sum(axis=0) + 0.02, label="HVD Component Sum All Orders", linewidth=LINEWIDTH)

plt.legend()
if SAVEFIG: plt.savefig(f"03 Orders of Decomposition.{FORMATPLT}", format=FORMATPLT, dpi=DPI)
if SHOWPLT: plt.show()
plt.clf()


# Applying Savitzky-Golay Filter
breath_trim_savgol = sig.savgol_filter(breath_trim, window_length = 300, polyorder = 6)

# plot filtered array
plt.plot(x_axis_trim, breath_trim + 0.02, alpha=0.3, label="Pre-Filter", linewidth=LINEWIDTH)
plt.plot(x_axis_trim, breath_trim_savgol, label="Savitzky-Golay", linewidth=LINEWIDTH)

plt.legend()
if SAVEFIG: plt.savefig(f"04 After Savitzky-Golay.{FORMATPLT}", format=FORMATPLT, dpi=DPI)
if SHOWPLT: plt.show()



#------------------------------------------------------
# This Section Preforms Cross-Corellation Between the Measurement and HITRAN spectra


# Water -------------------------------
h2o = np.empty((2, 442498))

# load H2O spectrum from HITRAN
h2o[0] = np.loadtxt("HITRAN/SpectrMol_H2O.coef.txt", skiprows=6, usecols = 0)
h2o[1] = np.loadtxt("HITRAN/SpectrMol_H2O.coef.txt", skiprows=6, usecols = 1)

plt.plot(x_axis_trim, breath_trim_savgol, alpha=0.7, label="Breath Spectrum", linewidth=LINEWIDTH)
plt.plot(h2o[0], h2o[1], label="H2O Spectrum", linewidth=LINEWIDTH)

plt.legend()
if SAVEFIG: plt.savefig(f"05 Breath vs Water Spectra.{FORMATPLT}", format=FORMATPLT, dpi=DPI)
if SHOWPLT: plt.show()


# Preform Cross-Corellation and Add to the Plot
#cross_correlation = np.correlate(breath_trim_savgol, h2o[1], mode='same')
#h2o_corr = np.correlate(h2o[1], h2o[1], mode="same")

#plt.plot(x_axis_trim, cross_correlation - 0.02, alpha=0.5, label="Correlation")
#plt.plot(h2o[0], h2o_corr, label="Water self-corr test", linewidth=LINEWIDTH)
#plt.plot(x_axis_trim, breath_trim_savgol, alpha=0.7, label="Breath Spectrum", linewidth=LINEWIDTH)
#plt.plot(h2o[0], h2o[1], label="H2O Spectrum", linewidth=LINEWIDTH)

plt.legend()
if SAVEFIG: plt.savefig(f"06 Breath vs Water Spectra + Corr.{FORMATPLT}", format=FORMATPLT, dpi=DPI)
if SHOWPLT: plt.show()
plt.clf()


# Carbon Dioxide ----------------------
co2 = np.empty((2, 442498))

# load CO2 spectrum from HITRAN
co2[0] = np.loadtxt("HITRAN/SpectrMol_CO2.coef.txt", skiprows=6, usecols = 0)
co2[1] = np.loadtxt("HITRAN/SpectrMol_CO2.coef.txt", skiprows=6, usecols = 1)

plt.plot(x_axis_trim, breath_trim_savgol, alpha=0.7, label="Breath Spectrum", linewidth=LINEWIDTH)
plt.plot(co2[0], co2[1], label="CO2 Spectrum", linewidth=LINEWIDTH)

plt.legend()
if SAVEFIG: plt.savefig(f"07 Breath vs CO2 Spectra.{FORMATPLT}", format=FORMATPLT, dpi=DPI)
if SHOWPLT: plt.show()




'''
#------------------------------------------------------
# This Section Interpolates the Spectrum to obtain data points that lie on the h2o[0] grid for cleaner corellation

breath_interp = np.interp(h2o[0], x_axis_trim, breath_trim, left=0, right=0)

plt.plot(h2o[0], breath_interp, label="interp", linewidth=LINEWIDTH)
plt.plot(h2o[0], h2o[1], label="water", linewidth=LINEWIDTH)


#print(npcorr)
#npcorr2 = np.correlate(breath_interp, h2o[1], mode='same')
#print(npcorr2)


#plt.plot(npcorr2, label="interp")

plt.legend()
plt.show()
'''
