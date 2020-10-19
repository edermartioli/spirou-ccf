# -*- coding: iso-8859-1 -*-
"""
    Created on October 18 2020
    
    Description:  library to handle spectral lines to build CCF mask
    
    @author: Eder Martioli, <martioli@iap.fr>
    
    Institut d'Astrophysique de Paris, France.
    
    """

__version__ = "1.0"

__copyright__ = """
    Copyright (c) ...  All rights reserved.
    """
import os,sys

from scipy import constants

import astropy.io.fits as fits

import matplotlib.pyplot as plt
import matplotlib

import numpy as np
from copy import copy, deepcopy

import spiroulib

from astropy import units as u

from astropy.convolution import Gaussian1DKernel, convolve

from scipy.signal import find_peaks
from scipy.interpolate import interp1d

from scipy.optimize import curve_fit
from astropy.modeling.models import Voigt1D, Gaussian1D

from astropy.table import Table

import warnings

def thermal_ccf(v=[], v0=0., line_depth=1.0, atomic_number=1, Teff=5800, plot=False) :

    if len(v) == 0 :
        v_min, v_max, v_step = -100, 100, 0.5
        v = np.arange(v_min, v_max+v_step, v_step)
    
    species_mass = 1.66053906660e-27 * atomic_number
    kb = 1.38064852e-23 # m2 kg s-2 K-1
    
    m2_in_km2 = 1e-6
    
    k_over_m =  kb * m2_in_km2 / species_mass
    
    mean_vel = np.sqrt(2. * k_over_m * Teff)
    
    exponent = (v - v0) / mean_vel

    profile = 1. - line_depth * np.exp(- exponent*exponent)

    if plot :
        plt.plot(v,profile)
        plt.xlabel(r"velocity [km/s]")
        plt.ylabel(r"relative flux")
        plt.show()
    
    line_profile = {}

    line_profile["type"] = "thermal"
    line_profile["v0"] = v0
    line_profile["line_depth"] = line_depth
    line_profile["atomic_number"] = atomic_number
    line_profile["Teff"] = Teff
    line_profile["fwhm"] = 2.3548 * np.sqrt(k_over_m * Teff)

    line_profile["flux"] = profile
    line_profile["vels"] = v

    return line_profile


def natural_ccf(v=[], v0=0., line_depth=1.0, wavelength=1000., plot=False) :
    
    if len(v) == 0 :
        v_min, v_max, v_step = -100, 100, 0.5
        v = np.arange(v_min, v_max+v_step, v_step)
    
    fwhm = 353.801 / wavelength
    
    profile = 1. - line_depth / (4.*((v-v0)/fwhm)*((v-v0)/fwhm) + 1)
    
    if plot :
        plt.plot(v,profile)
        plt.xlabel(r"velocity [km/s]")
        plt.ylabel(r"relative flux")
        plt.show()
    
    line_profile = {}
    
    line_profile["type"] = "natural"
    line_profile["v0"] = v0
    line_profile["line_depth"] = line_depth
    line_profile["wavelength"] = wavelength
    line_profile["fwhm"] = fwhm

    line_profile["flux"] = profile
    line_profile["vels"] = v

    return line_profile


def rotational_ccf(v=[], v0=0., line_depth=1.0, vsini=1., plot=False) :
    
    def rot_prof (v, v0, d, vsini) :
        profile = np.ones_like(v)
        mask = ((v - v0)*(v - v0)/(vsini * vsini)) < 1.
        profile[mask] = 1. - d * np.sqrt(1. - (v[mask] - v0)*(v[mask] - v0)/(vsini * vsini))
        return profile
    
    if len(v) == 0 :
        v_min, v_max, v_step = -100, 100, 0.5
        v = np.arange(v_min, v_max+v_step, v_step)
    
    profile = rot_prof(v, v0, line_depth, vsini)
    
    line_profile = {}

    line_profile["type"] = "rotational"
    line_profile["v0"] = v0
    line_profile["line_depth"] = line_depth
    line_profile["vsini"] = vsini
    line_profile["fwhm"] = np.sqrt(vsini)

    line_profile["flux"] = profile
    line_profile["vels"] = v

    if plot :
        plt.plot(v, profile)
        vpoints = np.array([v0 - line_profile["fwhm"]/2, v0 + line_profile["fwhm"]/2])
        plt.plot(vpoints,rot_prof(vpoints, v0, line_depth, vsini),'o')
        plt.xlabel(r"velocity [km/s]")
        plt.ylabel(r"relative flux")
        plt.show()

    return line_profile


# Get model CCF
def get_rot_ccf(v_min=-100, v_max=100, v_step=0.5, resolution=70000, wlc=1600., vsini=0., teff=3500, plot=False) :
    loc = {}
    loc['v_min'] = v_min
    loc['v_max'] = v_max
    loc['v_step'] = v_step
    
    # calculate profiles for wider range due to edge effects in convolution
    vels_tmp = np.arange(v_min-10, v_max+v_step+10, v_step)
    loc['rot_ccf'] = rotational_ccf(v=vels_tmp, vsini=vsini)

    fwhm_vel = (constants.c / 1000.) / resolution
    sig_vel = fwhm_vel / (2*np.sqrt(2*np.log(2)))
    g = Gaussian1DKernel(stddev=sig_vel/v_step)
    # Convolve data
    convolved_flux = convolve(loc['rot_ccf']['flux'], g)
    
    out = vels_tmp >= v_min
    out &= vels_tmp <= v_max

    loc['flux'] = convolved_flux[out]
    loc['vels'] = vels_tmp[out]

    half_maximum = (np.max(loc['flux']) + np.min(loc['flux']))/2
    
    positive = loc['vels'] > 0
    negative = loc['vels'] < 0
    
    pos_idx = np.argmin(np.abs(loc['flux'][positive] - half_maximum))
    neg_idx = np.argmin(np.abs(loc['flux'][negative] - half_maximum))

    loc['fwhm'] = np.abs(loc['vels'][positive][pos_idx] - loc['vels'][negative][neg_idx])

    if plot :
        font = {'size': 16}
        matplotlib.rc('font', **font)
        
        higres_vel = np.arange(v_min, v_max+v_step, v_step/5.)
        
        Fe_ccf = thermal_ccf(v=higres_vel, atomic_number=26, Teff=teff)
        plt.plot(Fe_ccf["vels"],Fe_ccf["flux"], "--", lw=1.2, color="green", label=r"Fe (Z=26) T=3500 K")
        Fe_ccf = thermal_ccf(v=higres_vel, atomic_number=26, Teff=9000)
        #plt.plot(Fe_ccf["vels"],Fe_ccf["flux"], ":", lw=0.6, color="green", label=r"Iron (Z=26) T=9000 K")

        C_ccf = thermal_ccf(v=higres_vel, atomic_number=6, Teff=teff)
        #plt.plot(C_ccf["vels"],C_ccf["flux"], "--", lw=1.2, color="blue", label=r"Carbon (Z=6) T=3500 K")
        C_ccf = thermal_ccf(v=higres_vel, atomic_number=6, Teff=9000)
        #plt.plot(C_ccf["vels"],C_ccf["flux"], ":", lw=0.6, color="blue", label=r"Carbon (Z=6) T=9000 K")

        He_ccf = thermal_ccf(v=higres_vel, atomic_number=2, Teff=teff)
        plt.plot(He_ccf["vels"],He_ccf["flux"], "--", lw=0.6, color='red' , label=r"He (Z=2) T=3500 K")

        plt.plot(loc['vels'], loc['rot_ccf']['flux'][out], ":", lw=2, label=r"Rotation $v$sin$(i)$={0:.1f} km/s".format(vsini))
        plt.plot(loc['vels'], loc['flux'], "-", lw=3, color='k', label=r"Instrument (R={0:.0f}) * Rotation".format(resolution))

        plt.legend(fontsize=10, loc='lower right')
        plt.ylabel(r"Flux")
        plt.xlabel(r"Velocity [km/s]")
        plt.show()

    f_ccf = interp1d(loc['vels'], loc['flux'], kind='cubic')
    loc['f_ccf'] = f_ccf

    return loc


def detect_lines_in_template(template_spectrum, ccf_width, ccf_step, resolution, vsini, min_n_points=1000, normalize=True, linemask=None, plot=False) :
    
    loc = {}
    loc['ccf_width'] = ccf_width
    loc['ccf_step'] = ccf_step
    loc['resolution'] = resolution
    loc['ccf_width'] = ccf_width
    loc['vsini'] = vsini
    loc['min_n_points'] = min_n_points
    
    # Load spectral orders for spirou
    spirou_orders = spiroulib.spirou_order_mask()

    line_wlc = []
    line_weight, line_xcorr = [], []
    
    flat_line_wlc = np.array([])
    flat_weight_wlc = np.array([])
    
    out_wl, out_flux, out_fluxerr = [], [], []
    
    for order in range(len(spirou_orders['orders'])) :
        #for order in range(28,38) :
        #for order in range(41,45) :
        
        # mask NaNs in template
        mask = ~np.isnan(template_spectrum['flux'][order])

        # get order masked data
        wl, flux, fluxerr = template_spectrum['wl'][order][mask], template_spectrum['flux'][order][mask], template_spectrum['fluxerr'][order][mask]
        
        # proceed if data vector is > min_n_points
        if len(wl) < min_n_points :
            line_wlc.append(np.array([]))
            line_weight.append(np.array([]))
            line_xcorr.append(np.array([]))
            
            out_wl.append(np.array([]))
            out_flux.append(np.array([]))
            out_fluxerr.append(np.array([]))
            continue
        
        
        if linemask != None :
            # filter-out lines outside order range
            linemask_wl_filter = linemask['LL_MASK_CTR'] > wl[0]
            linemask_wl_filter &= linemask['LL_MASK_CTR'] < wl[-1]
            ll_mask_d = linemask['LL_MASK_D'][linemask_wl_filter]
            ll_mask_ctr = linemask['LL_MASK_CTR'][linemask_wl_filter]
            w_mask = linemask['W_MASK'][linemask_wl_filter]
        
        
        # Detect continuum and normalize order spectrum. It's good to do it even if already normalized
        if normalize:
            warnings.simplefilter('ignore', np.RankWarning)

            continuum = spiroulib.fit_continuum(wl, flux, function='polynomial', order=5,
                                      nit=5, rej_low=1.0, rej_high=2.5, grow=1,
                                      med_filt=1, percentile_low=0.,
                                      percentile_high=100.,min_points=10,
                                      xlabel="", ylabel="", plot_fit=False, verbose=False)
            flux /= continuum
            fluxerr /= continuum

        out_wl.append(wl)
        out_flux.append(flux)
        out_fluxerr.append(fluxerr)

        # Calculate order central wavelength
        wlc = (wl[0] + wl[-1]) / 2.

        # Create a ccf from convolution of rotation profile and gaussian kernel fro inst. resolution
        ccf = get_rot_ccf(v_min=-ccf_width, v_max=ccf_width, v_step=ccf_step, resolution=resolution, wlc=wlc, vsini=vsini, plot=False)

        # Initialize cross-correlation vector
        xcorr = []
        
        for i in range(len(wl)) :
            # find wavelength window covering the same range in velocity space as the input CCF
            wl_min = wl[i] * (1.0 + ccf['vels'][1]/(constants.c/1000.))
            wl_max = wl[i] * (1.0 + ccf['vels'][-2]/(constants.c/1000.))
        
            # select data within range:
            wl_mask = wl > wl_min
            wl_mask &= wl < wl_max
        
            # get data within range
            wl_range, flux_range, fluxerr_range = wl[wl_mask], flux[wl_mask], fluxerr[wl_mask]
            vel_range = (wl_range - wl[i]) * (constants.c/1000.) / wl_range

            # calculate cross-correlation between spectrum and ccf
            xc = np.ma.corrcoef(ccf['f_ccf'](vel_range),flux_range).data[0,1]

            xcorr.append(xc)
        # cast xcorr array
        xcorr = np.array(xcorr)

        # run find_peaks algorithm to fin highest correlation values
        peaks, _ = find_peaks(xcorr)

        # select only peaks with positive correlation
        mask = xcorr[peaks] > 0
        mask &= flux[peaks] < 1.
        
        final_peaks_wl = wl[peaks][mask]
        final_peaks_xcorr = xcorr[peaks][mask]
        final_peaks_depth = flux[peaks][mask]
        
        if len(final_peaks_wl) :
            line_wlc.append(final_peaks_wl)
            line_weight.append(1. - final_peaks_depth)
            line_xcorr.append(final_peaks_xcorr)
        else :
            line_wlc.append(np.array([]))
            line_weight.append(np.array([]))
            line_xcorr.append(np.array([]))

        # calculate min to max amplitude of x-correlation function to normalize it
        xcorr_amp = np.max(xcorr) + np.abs(np.min(xcorr))
        
        dvel_min = ccf['fwhm'] / 10
        dwl =  dvel_min * wlc / (constants.c/1000.)
    
        repeated_peaks_wl, repeated_peaks_depth = [], []
        
        for j in range(len(final_peaks_wl)) :
            
            check_mask = flat_line_wlc >= final_peaks_wl[j] - dwl
            check_mask &= flat_line_wlc <= final_peaks_wl[j] + dwl

            # add peak only if it doesn't already exist in the mask
            if len(flat_line_wlc[check_mask]) == 0 :
                flat_line_wlc = np.append(flat_line_wlc, final_peaks_wl[j])
                flat_weight_wlc = np.append(flat_weight_wlc,  1. - final_peaks_depth[j])
            else :
                repeated_peaks_wl.append(final_peaks_wl[j])
                repeated_peaks_depth.append(final_peaks_depth[j])

        if plot :
            #plt.title("Spectral Order {0:d}".format(order))
            plt.plot(final_peaks_wl, final_peaks_depth, "x", color="k")
            plt.plot(final_peaks_wl, 1. - final_peaks_depth, "o", color="olive", alpha=0.6)
            plt.plot(wl, xcorr/xcorr_amp + 1.0, ':',linewidth=0.6, color="grey")
            if len(repeated_peaks_wl) :
                plt.plot(repeated_peaks_wl, repeated_peaks_depth, "o", color="red", alpha=0.6)
            #plt.scatter(wl, flux, marker='.', s=10., edgecolors='tab:blue', facecolors='none', lw=.5)
            plt.errorbar(wl, flux, yerr=fluxerr, fmt='.', alpha=0.5)
            #plt.show()

    loc['wlc'] = line_wlc
    loc['weight'] = line_weight
    loc['xcorr'] = line_xcorr

    loc['flat_wlc'] = flat_line_wlc
    loc['flat_weight'] = flat_weight_wlc

    loc['wl'] = out_wl
    loc['flux'] = out_flux
    loc['fluxerr'] = out_fluxerr

    if plot :
        plt.ylabel("flux")
        plt.xlabel("wavelength (nm)")
        plt.show()

    return loc

def voigt_hjerting(x) :
    #Approximation - Eq. (24) in Thorsten Tepper Garcia (2006) Mon. Not. R. Astron. Soc. 369, 2025–2035 (2006)
    fac1 = 1. / (2. * x**2)
    fac2 = (4 * x**2 + 3)*(x**2 + 1)
    fac3 = (2*x**2 + 3) / (x**2)
    K =  fac1 * (fac2 * np.exp(-x**2) - fac3 * np.sinh(x**2))
    H = -2.*K*np.exp(-x**2)/np.sqrt(np.pi)
    print("fac1->",fac1)
    print("fac2->",fac2)
    print("fac3->",fac3)
    print("K->",K)
    print("H->",H)
    return H

def doppler_width(wlc, T, Z) :
    dv = (1./11.) * np.sqrt(T/Z)
    dwl = dv * wlc / (constants.c/1000)
    return dwl, dv


def load_VALD_catalog(filename) :
    
    
    #element    wave_A    wave_nm    loggf    lower_state_eV    lower_state_cm1    lower_j    upper_state_eV    upper_state_cm1    upper_j    upper_g    lande_lower    lande_upper    spectrum_transition_type    turbospectrum_rad    rad    stark    waals    waals_single_gamma_format    turbospectrum_fdamp    spectrum_fudge_factor    theoretical_depth    theoretical_ew    lower_orbital_type    upper_orbital_type    molecule    spectrum_synthe_isotope    ion    spectrum_moog_species    turbospectrum_species    width_species    reference_code    spectrum_support    turbospectrum_support    moog_support    width_support    synthe_support    sme_support

    
    loc = {}
    loc["filename"] = filename
    
    loc["element"], loc["redox"] = [], []
    loc["wave_nm"] = []
    loc["loggf"] = []
    loc["lower_state_eV"], loc["upper_state_eV"] = [], []
    loc["lower_j"], loc["upper_j"], loc["upper_g"] = [], [], []
    loc["lande_lower"], loc["lande_upper"] = [], []
    loc["spectrum_transition_type"] = []
    loc["rad"], loc["stark"], loc["waals"], loc["waals_single_gamma_format"] = [], [], [], []
    loc["theoretical_depth"], loc["theoretical_ew"] = [], []
    loc["lower_orbital_type"], loc["upper_orbital_type"] = [], []
    loc["atomic_num"], loc["atomic_num_code"] = [], []
    loc["molecule"] = []
    
    f = open(filename, 'r')
    
    for line in f:
        if line[0] != '#':
            cols = line.split()

            loc["element"].append(cols[0])
            loc["redox"].append(int(cols[1]))
            loc["wave_nm"].append(float(cols[3]))
            loc["loggf"].append(float(cols[4]))
            loc["lower_state_eV"].append(float(cols[5]))
            loc["upper_state_eV"].append(float(cols[8]))
            loc["lower_j"].append(float(cols[7]))
            loc["upper_j"].append(float(cols[10]))
            loc["upper_g"].append(float(cols[11]))
            loc["lande_lower"].append(float(cols[12]))
            loc["lande_upper"].append(float(cols[13]))
            loc["spectrum_transition_type"].append(cols[14])
            loc["rad"].append(float(cols[16]))
            loc["stark"].append(float(cols[17]))
            loc["waals"].append(float(cols[18]))
            loc["waals_single_gamma_format"].append(cols[19])
            loc["theoretical_depth"].append(float(cols[22]))
            loc["theoretical_ew"].append(float(cols[23]))
            loc["lower_orbital_type"].append(cols[24])
            loc["upper_orbital_type"].append(cols[25])
            if cols[26] == "T" :
                loc["molecule"].append(True)
            else :
                loc["molecule"].append(False)
            loc["atomic_num"].append(float(cols[29]))
            loc["atomic_num_code"].append(cols[30])
    
    loc["element"] = np.array(loc["element"])
    loc["redox"] = np.array(loc["redox"])
    loc["wave_nm"] = np.array(loc["wave_nm"])
    loc["wave_nm_vacuum"] = convert_air_to_vacuum_wl(loc["wave_nm"], air_density=1.0)
    loc["loggf"] = np.array(loc["loggf"])
    loc["lower_state_eV"] = np.array(loc["lower_state_eV"])
    loc["upper_state_eV"] = np.array(loc["upper_state_eV"])
    loc["lower_j"] = np.array(loc["lower_j"])
    loc["upper_j"] = np.array(loc["upper_j"])
    loc["upper_g"] = np.array(loc["upper_g"])
    loc["lande_lower"] = np.array(loc["lande_lower"])
    loc["lande_upper"] = np.array(loc["lande_upper"])
    loc["spectrum_transition_type"] = np.array(loc["spectrum_transition_type"])
    loc["rad"] = np.array(loc["rad"])
    loc["stark"] = np.array(loc["stark"])
    loc["waals"] = np.array(loc["waals"])
    loc["waals_single_gamma_format"] = np.array(loc["waals_single_gamma_format"])
    loc["theoretical_depth"] = np.array(loc["theoretical_depth"])
    loc["theoretical_ew"] = np.array(loc["theoretical_ew"])
    loc["lower_orbital_type"] = np.array(loc["lower_orbital_type"])
    loc["upper_orbital_type"] = np.array(loc["upper_orbital_type"])
    loc["atomic_num"] = np.array(loc["atomic_num"])
    loc["molecule"] = np.array(loc["molecule"])
    loc["atomic_num_code"] = np.array(loc["atomic_num_code"])
    
    return loc


def nrefrac(wavelength, density=1.0):
   """Calculate refractive index of air from Cauchy formula.

   Input: wavelength in nm, density of air in amagat (relative to STP,
   e.g. ~10% decrease per 1000m above sea level).
   Returns N = (n-1) * 1.e6.
   """

   # The IAU standard for conversion from air to vacuum wavelengths is given
   # in Morton (1991, ApJS, 77, 119). For vacuum wavelengths (VAC) in
   # Angstroms, convert to air wavelength (AIR) via:

   #  AIR = VAC / (1.0 + 2.735182E-4 + 131.4182 / VAC^2 + 2.76249E8 / VAC^4)

   wl2inv = (1.e3/wavelength)**2
   refracstp = 272.643 + 1.2288 * wl2inv  + 3.555e-2 * wl2inv**2
   return density * refracstp

def convert_vacuum_to_air_wl(vacuum_wavelength, air_density=1.0) :
    air_wavelength = vacuum_wavelength / ( 1. + 1.e-6 * nrefrac(vacuum_wavelength, density=air_density))
    return air_wavelength

def convert_air_to_vacuum_wl(air_wavelength, air_density=1.0) :
    vacuum_wavelength = air_wavelength * ( 1. + 1.e-6 * nrefrac(air_wavelength, density=air_density))
    return vacuum_wavelength


def search_strong_lines (vald_database, wl0=950, wlf=2500, Teff=3500, element="He", redox=0, plot=False) :
    # Load VALD database
    vald = load_VALD_catalog(vald_database)
    
    mask = vald["element"] == element
    if redox :
        mask = vald["redox"] == redox

    mask &= vald['wave_nm_vacuum'] > wl0
    mask &= vald['wave_nm_vacuum'] < wlf

    element = vald['element'][mask]
    redox = vald['redox'][mask]
    wlc = vald['wave_nm_vacuum'][mask]
    loggf = vald['loggf'][mask]
    upper_state_eV = vald['upper_state_eV'][mask]
    lower_state_eV = vald['lower_state_eV'][mask]

    atomic_num = vald['atomic_num'][mask]
    molecule = vald['molecule'][mask]
    atomic_num_code = vald['atomic_num_code'][mask]
    atomic_mass = atomic_num
    for i in range(len(atomic_mass)) :
        if molecule[i] :
            atomic_mass[i] = float(atomic_num_code[i].split(".")[0][0:2]) + float(atomic_num_code[i].split(".")[0][2:4])

    Energy = upper_state_eV - lower_state_eV
    kT = 8.617333262e-5 * Teff # eV
    
     # Below was an attempt to calculate line opacity, but in fact since we just need
    # to compare opacities to get the one with highest contribution for this line, I have
    # removed terms related to lambda (which are all the same) and line profiles, which
    # will be analyzed in a second step
    #gamma = vald_rad + vald_stark + vald_waals
    #alpha = gamma / (4. * np.pi * dwl_doppler)
    #Hfunc = voigt_hjerting(alpha)
    # Eq 1 in Piskunov et al. (1995)
    #line_opacity = vald_loggf + (2*np.log(vald_wlc)) - (Energy/kT) + np.log(Hfunc)
    line_opacity = loggf - (Energy/kT)

    sort = np.argsort(line_opacity)

    for i in range(len(wlc[sort])) :
        dwl_doppler, dv_doppler = doppler_width(wlc[sort][i], Teff, atomic_mass[sort][i])
        print(element[sort][i], redox[sort][i], wlc[sort][i], line_opacity[sort][i], dv_doppler)


def merge_wl_ranges(wlranges) :
    #This routine takes a list of spectral ranges
    # and merge them into the minimum number of ranges
    
    # initialize final vector of spectral ranges
    out_wlranges = []
    # initialize current wl0 and wlf
    current_wl0, current_wlf = wlranges[0][0], wlranges[0][1]
    # merge overlapping ranges
    for r in wlranges :
        if r[0] <= current_wlf :
            current_wlf = r[1]
        else :
            out_wlranges.append([current_wl0, current_wlf])
            current_wl0 = r[0]
            current_wlf = r[1]
    # append last range
    out_wlranges.append([current_wl0, current_wlf])
    return out_wlranges


def fit_multi_gaussian(x, y, weights=None, guess=None, return_fit=True, return_uncertainties=False):
    """
    Fit multiple gaussian to the data "y" at positions "x", points can be
    weighted by "weights" and an initial guess for the gaussian parameters

    :param x: numpy array (1D), the x values for the gaussian
    :param y: numpy array (1D), the y values for the gaussian
    :param weights: numpy array (1D), the weights for each y value
    :param guess: list of floats, the initial guess for the multiple guassian
                  fit parameters in the following order:

                  [offset from 0 (in y-direction), a_0, ctr_0, fwhm_0, a_1, ctr_1, fwhm_1,..., a_n, ctr_n, fwhm_n]

    :param return_fit: bool, if True also calculates the fit values for x
                       i.e. yfit = gauss_function(x, *pfit)

    :param return_uncertainties: bool, if True also calculates the uncertainties
                                 based on the covariance matrix (pcov)
                                 uncertainties = np.sqrt(np.diag(pcov))

    :return pfit: numpy array (1D), the fit parameters in the
                  following order:

                [[amplitude, center, fwhm, offset from 0 (in y-direction)],[],..,[]]

    :return yfit: numpy array (1D), the fit y values, i.e. the gaussian values
                  for the fit parameters, only returned if return_fit = True

    """

    # if we don't have weights set them to be all equally weighted
    if weights is None:
        weights = np.ones(len(x))
    weights = np.sqrt(1.0 / weights)
    # if we aren't provided a guess, make a guess assuming a single gaussian
    if guess is None:
        guess = np.array([0, np.max(y), np.mean(y), np.std(y)])
    
    # calculate the fit using curve_fit to the function "gauss_function"
    pfit, pcov = curve_fit(multi_gauss_function, x, y, p0=guess, sigma=weights,
                               absolute_sigma=True)

    if return_fit and return_uncertainties:
        # calculate the fit parameters
        yfit = multi_gauss_function(x, *pfit)
        
        # work out the normalisation constant and error calculation
        efit = []
        if( len(y) > len(guess) ) and pcov is not None:
            s_sq = ((yfit)**2).sum() / (len(y) - len(guess))
            pcov *= s_sq
        for j in range(len( pcov)):
            efit.append(pcov[j][j]**0.5)

        # return pfit, yfit and efit
        return pfit, yfit, efit
    # if just return fit
    elif return_fit:
        # calculate the fit parameters
        yfit = multi_gauss_function(x, *pfit)
        # return pfit and yfit
        return pfit, yfit
    # if return uncertainties
    elif return_uncertainties:
        # calculate the fit parameters
        yfit = multi_gauss_function(x, *pfit)
        # work out the normalisation constant
        chis, _ = chisquare(y, f_exp=yfit)
        norm = chis / (len(y) - len(guess))
        # calculate the fit uncertainties based on pcov
        efit = np.sqrt(np.diag(pcov)) * np.sqrt(norm)
        # return pfit and efit
        return pfit, efit
    # else just return the pfit
    else:
        # return pfit
        return pfit


def multi_gauss_function(x, *pars):
    """
    A standard 1D gaussian function (for fitting against)]=

    :param x: numpy array (1D), the x data points
    :param dc: float, the constant level below all gaussians
    :param pars: array of gaussian parameters as follows:
    [dc, a_0, x0_0, sigma_0, ..., a_n, x0_n, sigma_n], where:
                :param a: float, the amplitude
                :param x0: float, the mean of the gaussian
                :param sigma: float, the standard deviation (FWHM) of the gaussian
    :return gauss: numpy array (1D), size = len(x), the output gaussian
    """
    
    multi_gauss = pars[0]
    ngauss = int((len(pars) - 1) / 3)
    for i in range(ngauss) :
        multi_gauss += pars[1 + 3 * i] * np.exp(-0.5 * ((x - pars[2 + 3 * i]) / pars[3 + 3 * i]) ** 2)
    
    return multi_gauss


def atoms() :
    atoms = {"Hydrogen": ["H", 1],"Helium": ["He", 2],"Lithium": ["Li", 3],"Beryllium": ["Be", 4],"Boron": ["B", 5],"Carbon": ["C", 6],"Nitrogen": ["N", 7],"Oxygen": ["O", 8],"Fluorine": ["F", 9],"Neon": ["Ne", 10],"Sodium": ["Na", 11],"Magnesium": ["Mg", 12],"Aluminium": ["Al", 13],"Silicon": ["Si", 14],"Phosphorus": ["P", 15],"Sulfur": ["S", 16],"Chlorine": ["Cl", 17],"Argon": ["Ar", 18],"Potassium": ["K", 19],"Calcium": ["Ca", 20],"Scandium": ["Sc", 21],"Titanium": ["Ti", 22],"Vanadium": ["V", 23],"Chromium": ["Cr", 24],"Manganese": ["Mn", 25],"Iron": ["Fe", 26],"Cobalt": ["Co", 27],"Nickel": ["Ni", 28],"Copper": ["Cu", 29],"Zinc": ["Zn", 30],"Gallium": ["Ga", 31],"Germanium": ["Ge", 32],"Arsenic": ["As", 33],"Selenium": ["Se", 34],"Bromine": ["Br", 35],"Krypton": ["Kr", 36],"Rubidium": ["Rb", 37],"Strontium": ["Sr", 38],"Yttrium": ["Y", 39],"Zirconium": ["Zr", 40],"Niobium": ["Nb", 41],"Molybdenum": ["Mo", 42],"Technetium": ["Tc", 43],"Ruthenium": ["Ru", 44],"Rhodium": ["Rh", 45],"Palladium": ["Pd", 46],"Silver": ["Ag", 47],"Cadmium": ["Cd", 48],"Indium": ["In", 49],"Tin": ["Sn", 50],"Antimony": ["Sb", 51],"Tellurium": ["Te", 52],"Iodine": ["I", 53],"Xenon": ["Xe", 54],"Caesium": ["Cs", 55],"Barium": ["Ba", 56],"Lanthanum": ["La", 57],"Cerium": ["Ce", 58],"Praseodymium": ["Pr", 59],"Neodymium": ["Nd", 60],"Promethium": ["Pm", 61],"Samarium": ["Sm", 62],"Europium": ["Eu", 63],"Gadolinium": ["Gd", 64],"Terbium": ["Tb", 65],"Dysprosium": ["Dy", 66],"Holmium": ["Ho", 67],"Erbium": ["Er", 68],"Thulium": ["Tm", 69],"Ytterbium": ["Yb", 70],"Lutetium": ["Lu", 71],"Hafnium": ["Hf", 72],"Tantalum": ["Ta", 73],"Tungsten": ["W", 74],"Rhenium": ["Re", 75],"Osmium": ["Os", 76],"Iridium": ["Ir", 77],"Platinum": ["Pt", 78],"Gold": ["Au", 79],"Mercury": ["Hg", 80],"Thallium": ["Tl", 81],"Lead": ["Pb", 82],"Bismuth": ["Bi", 83],"Polonium": ["Po", 84],"Astatine": ["At", 85],"Radon": ["Rn", 86],"Francium": ["Fr", 87],"Radium": ["Ra", 88],"Actinium": ["Ac", 89],"Thorium": ["Th", 90],"Protactinium": ["Pa", 91],"Uranium": ["U", 92],"Neptunium": ["Np", 93],"Plutonium": ["Pu", 94],"Americium": ["Am", 95],"Curium": ["Cm", 96],"Berkelium": ["Bk", 97],"Californium": ["Cf", 98],"Einsteinium": ["Es", 99],"Fermium": ["Fm", 100],"Mendelevium": ["Md", 101],"Nobelium": ["No", 102],"Lawrencium": ["Lr", 103],"Rutherfordium": ["Rf", 104],"Dubnium": ["Db", 105],"Seaborgium": ["Sg", 106],"Bohrium": ["Bh", 107],"Hassium": ["Hs", 108],"Meitnerium": ["Mt", 109],"Darmstadtium": ["Ds", 110],"Roentgenium": ["Rg", 111],"Copernicium": ["Cn", 112],"Ununtrium": ["Uut", 113],"Flerovium": ["Fl", 114],"Ununpentium": ["Uup", 115],"Livermorium": ["Lv", 116],"Ununseptium": ["Uus", 117],"Ununoctium": ["Uuo", 118]}



def identify_detected_lines (loc, vald_database, min_vel_tol, Teff, linemask=None, verbose=False, plot=False) :
    
    # Load VALD database
    vald = load_VALD_catalog(vald_database)
    
    new_line_wlc = []
    new_line_depth = []
    new_line_sigma = []
    new_line_vald_wlc = []
    new_line_element = []
    new_line_redox = []
    new_line_atomic_num = []
    new_line_molecule = []
    new_line_atomic_mass = []
    new_line_doppler_v = []
    new_line_group_id = []
    new_line_rotinst_fwhm = []
    new_line_group_meanfluxerr = []
    new_line_group_noise = []
    new_line_group_mediandvel = []
    new_line_group_maddvel = []

    for order in range(len(loc['wl'])) :
        new_line_wlc.append(np.array([]))
        new_line_depth.append(np.array([]))
        new_line_sigma.append(np.array([]))
        new_line_vald_wlc.append(np.array([]))
        new_line_element.append(np.array([]))
        new_line_redox.append(np.array([]))
        new_line_atomic_num.append(np.array([]))
        new_line_molecule.append(np.array([]))
        new_line_atomic_mass.append(np.array([]))
        new_line_doppler_v.append(np.array([]))
        new_line_group_id.append(np.array([]))
        new_line_rotinst_fwhm.append(np.array([]))
        new_line_group_meanfluxerr.append(np.array([]))
        new_line_group_noise.append(np.array([]))
        new_line_group_mediandvel.append(np.array([]))
        new_line_group_maddvel.append(np.array([]))

    group_id = 0

    for order in range(len(loc['wl'])) :
    #for order in range(8,9) : # HeI lines
    #for order in range(34,36) :
    
        #wlmask = loc['wl'][order] > 1080
        #wlmask &= loc['wl'][order] < 1085.5
        #wl = loc['wl'][order][wlmask]
        #flux = loc['flux'][order][wlmask]
        #fluxerr = loc['fluxerr'][order][wlmask]

        wl = loc['wl'][order]
        flux = loc['flux'][order]
        fluxerr = loc['fluxerr'][order]
    
        # proceed if data vector is not empty
        if len(wl) < loc['min_n_points'] :
            continue
        
        if plot :
            plt.plot(wl, flux,':', lw=0.5, color="grey")

        # Calculate order central wavelength
        wlc = (wl[0] + wl[-1]) / 2.

        # Create a ccf from convolution of rotation profile and gaussian kernel fro inst. resolution
        ccf = get_rot_ccf(v_min=-loc['ccf_width'], v_max=loc['ccf_width'], v_step=loc['ccf_step'], resolution=loc['resolution'], wlc=wlc, vsini=loc['vsini'], plot=False)

        line_wlc = loc['wlc'][order]
        line_weight = loc['weight'][order]

        # initialize line ranges
        line_ranges = []
        
        # loop over each line in the mask to create spectral ranges around the lines
        for l in range(len(line_wlc)) :
            line_wl0 = line_wlc[l] * (1. - 1*ccf["fwhm"] / (constants.c / 1000.))
            line_wlf = line_wlc[l] * (1. + 1*ccf["fwhm"] / (constants.c / 1000.))
            line_ranges.append([line_wl0, line_wlf])
        
        # merge overlapping ranges into a single larger range
        if len(line_ranges) :
            merged_line_ranges = merge_wl_ranges(line_ranges)
        else :
            continue

        print("Processing order={0} wl0={1:.2f} wlf={2:.2f} nlines={3} number of ranges={4}".format(order, wl[0], wl[-1], len(line_wlc), len(merged_line_ranges)))

        # loop over each merged range
        for r in merged_line_ranges :
            group_id += 1
            
            # create wavelength mask to select lines within the range
            lines_in_range = line_wlc > r[0]
            lines_in_range &= line_wlc < r[1]
            
            # if a line mask is given include these lines in plot for comparison
            if linemask != None :
                linemask_wl_filter = linemask['LL_MASK_CTR'] > r[0]
                linemask_wl_filter &= linemask['LL_MASK_CTR'] < r[1]
                ll_mask_d = linemask['LL_MASK_D'][linemask_wl_filter]
                ll_mask_ctr = linemask['LL_MASK_CTR'][linemask_wl_filter]
                w_mask = linemask['W_MASK'][linemask_wl_filter]
        
            # select lines within range
            l_ctr, l_w = line_wlc[lines_in_range], line_weight[lines_in_range]
            l_bottom = (1.0 - l_w)
            
            # set number of lines within range
            nlines = len(l_ctr)
            
            vald_range = vald['wave_nm_vacuum'] > r[0]
            vald_range &= vald['wave_nm_vacuum'] < r[-1]

            nvald_lines = len(vald['wave_nm_vacuum'][vald_range])
            
            # create wavelength mask to select spectral data within range
            range_mask = wl > r[0]
            range_mask &= wl < r[1]

            # calculate weights as w = 1. / variance
            weights = 1. / (fluxerr[range_mask] * fluxerr[range_mask])

            # calculate mean flux error:
            mean_err = np.nanmean(fluxerr[range_mask])
            
            if verbose :
                print("Range: wl0={0:.1f} wlf={1:.1f} Nlines={2} NValdLines={3} mean_err={4:.3f}".format(r[0], r[1], nlines, nvald_lines, mean_err))

            # Use FWHM of CCF, but it needs to be carried out from loc
            fwhm = ccf['fwhm']
            ccf_sigma = fwhm / (2*np.sqrt(2*np.log(2)))
            rwlc = (r[0] + r[1])/2
            ccf_sigma_nm = ccf_sigma * rwlc / (constants.c / 1000.)

            if plot :
                plt.errorbar(wl[range_mask],flux[range_mask],yerr=fluxerr[range_mask], color="lightcoral",fmt='.', alpha=0.5)
                
                if linemask != None :
                    for i in range(len(ll_mask_ctr)) :
                        #print(i, ll_mask_ctr[i], w_mask[i], ccf_sigma_nm/5)
                        plt.bar(ll_mask_ctr[i], w_mask[i], ccf_sigma_nm/5, bottom=(1. - w_mask[i]), color="blue", alpha=0.6)

            # initialize empty array for guessed parameters
            guess = []
            # set guess for continuum as 1.0
            guess.append(1.0)
            for l in range(nlines) :
                # set guess for line depth as - mask weight
                guess.append(-l_w[l])
                # set guess for line center as input mask ctr
                guess.append(l_ctr[l])
                # set guess for line width based on the fwhm of CCF
                guess.append(ccf_sigma_nm)
            # cast guess as numpy array
            guess = np.array(guess)

            # Try to fit multiple gaussian to the range
            try :
                # perform multiple gaussian fit
                result, fit, efit = fit_multi_gaussian(wl[range_mask], flux[range_mask], weights=weights, guess=guess, return_fit=True, return_uncertainties=True)
                rms_residuals = np.std(flux[range_mask]-fit)
                
                # loop over each line in range
                for l in range(nlines) :
                    # get fit parameters for current line
                    fit_cont = result[0]
                    fit_amp = result[1 + 3 * l]
                    fit_ctr = result[2 + 3 * l]
                    fit_sig = result[3 + 3 * l]
                    
                    # calculate deviation between measured and predicted line center
                    ctr_err = np.abs(fit_ctr - l_ctr[l])
                    
                    # calculate reference fwhm in units of wavelength
                    ref_sigma_nm = ccf_sigma *  l_ctr[l] / (constants.c / 1000.)
                    
                    fit_sig_vel = (constants.c / 1000.) * fit_sig / l_ctr[l]
                    ctr_err_vel = (constants.c / 1000.) * ctr_err / l_ctr[l]
                    
                    ctr_mask = wl[range_mask] > fit_ctr - 2*ref_sigma_nm
                    ctr_mask &= wl[range_mask] < fit_ctr + 2*ref_sigma_nm

                    wl_ctr = wl[range_mask][ctr_mask]
                    flux_ctr = flux[range_mask][ctr_mask]

                    line_meas_depth = np.interp(fit_ctr, wl_ctr, flux_ctr)
                    
                    # calculate distribution of wavelength sampling to avoid regions with
                    # poor sampling
                    loc_mask = ~np.isnan(flux_ctr)
                    if len(wl_ctr[loc_mask]) > 3:
                        loc_dvel = (constants.c/1000.) * np.abs(wl_ctr[loc_mask][1:] - wl_ctr[loc_mask][:-1]) / fit_ctr
                        median_dvel = np.median(loc_dvel)
                        mad_dvel = np.median(np.abs(loc_dvel - median_dvel)) / 0.67449
                    else :
                        median_dvel, mad_dvel = constants.c, constants.c
                    #print("dvel=",median_dvel," km/s  sigdvel=", mad_dvel, " km/s")
                    
                    # only consider lines with delta lambda greater than fwhm and
                    # where fit amplitude is lower than the mean error / 4 within range and
                    #if ctr_err_vel < fwhm and  -fit_amp > mean_err/4 :
                    if ctr_err_vel < fwhm and median_dvel < 2.5:
                        # plot multiple gaussian fit model
                        if verbose :
                            print("GOOD line {0:.3f} nm: ctr_err={1:.1f} < {2:.1f} km/s; fit_amp={3:.3f}(w={4:.3f}) > flux_err={5:.3f}".format(l_ctr[l], ctr_err_vel, fwhm, -fit_amp, l_w[l], mean_err))
                        vald_match = np.abs(fit_ctr - vald['wave_nm_vacuum'][vald_range]) < ref_sigma_nm / 2
                        vald_wlc = vald['wave_nm_vacuum'][vald_range][vald_match]
                        vald_element = vald['element'][vald_range][vald_match]
                        vald_redox = vald['redox'][vald_range][vald_match]
                        vald_atomic_num = vald['atomic_num'][vald_range][vald_match]
                        vald_molecule = vald['molecule'][vald_range][vald_match]
                        vald_atomic_num_code = vald['atomic_num_code'][vald_range][vald_match]
                        
                        vald_loggf = vald['loggf'][vald_range][vald_match]
                        vald_rad = vald['rad'][vald_range][vald_match]
                        vald_stark = vald['stark'][vald_range][vald_match]
                        vald_waals = vald['waals'][vald_range][vald_match]
                        vald_upper_state_eV = vald['upper_state_eV'][vald_range][vald_match]
                        vald_lower_state_eV = vald['lower_state_eV'][vald_range][vald_match]

                        #print(vald_element,vald_atomic_num_code,vald_atomic_num_code.split(".")[0][0:2],vald_atomic_num_code.split(".")[0][2:4])
                        vald_atomic_mass = vald_atomic_num
                        for i in range(len(vald_wlc)) :
                            if vald_molecule[i] :
                                vald_atomic_mass[i] = float(vald_atomic_num_code[i].split(".")[0][0:2]) + float(vald_atomic_num_code[i].split(".")[0][2:4])

                        dwl_doppler, dv_doppler = doppler_width(wlc, Teff, vald_atomic_mass)
                        Energy = vald_upper_state_eV - vald_lower_state_eV
                        kT = 8.617333262e-5 * Teff # eV
                    
                        # Below was an attempt to calculate line opacity, but in fact since we just need
                        # to compare opacities to get the one with highest contribution for this line, I have
                        # removed terms related to lambda (which are all the same) and line profiles, which
                        # will be analyzed in a second step
                        #gamma = vald_rad + vald_stark + vald_waals
                        #alpha = gamma / (4. * np.pi * dwl_doppler)
                        #Hfunc = voigt_hjerting(alpha)
                        # Eq 1 in Piskunov et al. (1995)
                        #line_opacity = vald_loggf + (2*np.log(vald_wlc)) - (Energy/kT) + np.log(Hfunc)
                        line_opacity = vald_loggf - (Energy/kT)
                        
                        #print(vald['element'][vald_range][vald_match])
                        #print(line_opacity)

                        if np.all(np.isnan(line_opacity)) or len(line_opacity) == 0 :
                            best_line = np.argmin(np.abs(fit_ctr - vald_wlc))
                        else :
                            best_line = np.argmax(line_opacity)
                        
                        #vald_atomic_mass = 0.
                        line_accuracy = fit_ctr - vald_wlc[best_line]
                        
                        new_line_wlc[order] = np.append(new_line_wlc[order], fit_ctr)
                        new_line_depth[order] = np.append(new_line_depth[order], 1.0 - line_meas_depth)
                        new_line_sigma[order] = np.append(new_line_sigma[order], fit_sig_vel)
                        new_line_vald_wlc[order] = np.append(new_line_vald_wlc[order], vald_wlc[best_line])
                        new_line_element[order] = np.append(new_line_element[order], vald_element[best_line])
                        new_line_redox[order] = np.append(new_line_redox[order], vald_redox[best_line])
                        new_line_atomic_num[order] = np.append(new_line_atomic_num[order], vald_atomic_num[best_line])
                        new_line_molecule[order] = np.append(new_line_molecule[order], vald_molecule[best_line])
                        new_line_atomic_mass[order] = np.append(new_line_atomic_mass[order], vald_atomic_mass[best_line])
                        new_line_doppler_v[order] = np.append(new_line_doppler_v[order], dv_doppler[best_line])
                        new_line_group_id[order] = np.append(new_line_group_id[order], group_id)
                        new_line_rotinst_fwhm[order] = np.append(new_line_rotinst_fwhm[order], fwhm)
                        new_line_group_meanfluxerr[order] = np.append(new_line_group_meanfluxerr[order], mean_err)
                        new_line_group_noise[order] = np.append(new_line_group_noise[order], rms_residuals)
                        
                        new_line_group_mediandvel.append(median_dvel)
                        new_line_group_maddvel.append(mad_dvel)

                        # plot ACCEPTED line
                        if plot :
                            plt.plot(wl[range_mask], fit, color="blue", lw=2, alpha=0.7)
                            bottom = line_meas_depth
                            plt.bar(fit_ctr, 1.0 - line_meas_depth, ref_sigma_nm / 5., bottom=bottom, color="green", alpha=0.5)
                            plt.scatter([vald_wlc[best_line]], [1.0], marker="|", lw=2, color='k')
                            plt.text(vald_wlc[best_line], 1.04, r"{0} {1}".format(vald_element[best_line], vald_redox[best_line]), color="k",fontsize=9,horizontalalignment='center',verticalalignment='center',rotation='vertical')
                    else :
                        if verbose :
                            print("BAD line {0:.3f} nm: ctr_err={1:.3f} nm; fit_amp={2:.3f}(w={3:.3f}) > flux_err={4:.3f}; fit_sig={5:.3f} km/s".format(l_ctr[l], ctr_err_vel, np.abs(fit_amp), l_w[l], mean_err, fit_sig_vel))
                        
                        # plot REJECTED line
                        if plot :
                            plt.plot(wl[range_mask], fit, color="grey", lw=2, alpha=0.7)
                            bottom = (1.0 - np.abs(fit_amp))
                            plt.bar(fit_ctr, np.abs(fit_amp), ref_sigma_nm / 5., bottom=bottom, color="red", alpha=0.5)
                        pass
            except :
                # print "fit failed" when line is rejected because fit didn't succeed
                if verbose :
                    print("BAD  line {0:.3f}: FIT FAILED".format(l_ctr[l]))
                pass

        if plot:
            plt.title("Spectral Order {0:d}".format(order))
            plt.ylabel(r"Flux")
            plt.xlabel(r"$\lambda$ [nm]")
            plt.show()

    catalog = {}

    catalog["wlc"] = new_line_wlc
    catalog["depth"] = new_line_depth
    catalog["sigma"] = new_line_sigma
    catalog["vald_wlc"] = new_line_vald_wlc
    catalog["element"] = new_line_element
    catalog["redox"] = new_line_redox
    catalog["atomic_num"] = new_line_atomic_num
    catalog["molecule"] = new_line_molecule
    catalog["atomic_mass"] = new_line_atomic_mass
    catalog["doppler_v"] = new_line_doppler_v
    catalog["group_id"] = new_line_group_id
    catalog["rotinst_fwhm"] = new_line_rotinst_fwhm
    catalog["mean_fluxerr"] = new_line_group_meanfluxerr
    catalog["group_noise"] = new_line_group_noise
    catalog["line_dvel"] = new_line_group_mediandvel
    catalog["line_sigdvel"] = new_line_group_maddvel

    return catalog


def generate_ccf_optimal_mask(catalog, mask_width = 1.0, sig_clip=1.0, min_atomic_mass=0., max_atom_mass=1e20, element="", outputmask="", include_orders_in_mask=False, min_fwhm = 0., max_fwhm=100., use_measured_wlc=True, plot=False) :

    outmask_order = np.array([])
    outmask_wl0 = np.array([])
    outmask_wlf = np.array([])
    outmask_weight = np.array([])

    total_number_of_lines = 0
    
    for order in range(len(catalog['vald_wlc'])) :
        print("Order {0} has {1} lines before sig-clip filtering..".format(order, len(catalog['vald_wlc'][order])))
        
        filter = catalog["depth"][order] > sig_clip * catalog["mean_fluxerr"][order]
        #filter &= catalog["depth"][order] > sig_clip * catalog["group_noise"][order]
        filter &= catalog["sigma"][order] > min_fwhm / (2.*np.sqrt(2.*np.log(2.)))
        filter &= catalog["sigma"][order] < max_fwhm / (2.*np.sqrt(2.*np.log(2.)))

        print("Order {0} has {1} lines before element/Z filtering..".format(order, len(catalog['vald_wlc'][order][filter])))

        if element != "" :
            filter_by_element = deepcopy((filter) & (catalog["element"][order] == element))
            
            if len(catalog["group_id"][order][filter]) :
            
                min_group_id = np.min(catalog["group_id"][order][filter])
                max_group_id = np.max(catalog["group_id"][order][filter])
                
                for group in range(int(min_group_id),int(max_group_id)+1) :
                    
                    filterbygroup1 = deepcopy((filter) & (catalog["group_id"][order] == group))
                    filterbygroup2 = deepcopy((filter_by_element) & (catalog["group_id"][order] == group))
                    
                    nlinesingroup = len(catalog["element"][order][filterbygroup1])
                    nlinesingroupforelem = len(catalog["element"][order][filterbygroup2])
                    
                    #print("Filtering element=",element, " group=",group, "Nlines=",nlinesingroup, "Nlines[",element,"]",nlinesingroupforelem)
                    
                    if nlinesingroup != nlinesingroupforelem :
                        filter &= catalog["group_id"][order] != group
            
            filter &= catalog["element"][order] == element
        else :
            filter_by_atomic_mass = deepcopy((filter) & (catalog["atomic_mass"][order] >= min_atomic_mass) & (catalog["atomic_mass"][order] <= max_atom_mass))

            if len(catalog["group_id"][order][filter]) :
                min_group_id = np.min(catalog["group_id"][order][filter])
                max_group_id = np.max(catalog["group_id"][order][filter])
                
                for group in range(int(min_group_id),int(max_group_id)+1) :
                    filterbygroup1 = deepcopy((filter) & (catalog["group_id"][order] == group))
                    filterbygroup2 = deepcopy((filter_by_atomic_mass) & (catalog["group_id"][order] == group))
                    
                    nlinesingroup = len(catalog["element"][order][filterbygroup1])
                    nlinesingroupforelem = len(catalog["element"][order][filterbygroup2])
                    
                    #print("Filtering ", min_atomic_mass, "< atomic_mass < ",max_atom_mass, " group=",group, "Nlines=",nlinesingroup, "Nlines[mass_range]",nlinesingroupforelem)
                    
                    if nlinesingroup != nlinesingroupforelem :
                        filter &= catalog["group_id"][order] != group
            
            filter &= (catalog["atomic_mass"][order] >= min_atomic_mass) & (catalog["atomic_mass"][order] <= max_atom_mass)

        if use_measured_wlc :
            wlcs = catalog["wlc"][order]
        else :
            wlcs = catalog["vald_wlc"][order]

        if plot :
            
            plt.plot(wlcs[filter], catalog["depth"][order][filter], '.', color="green", alpha=0.6)
            plt.plot(wlcs[~filter], catalog["depth"][order][~filter], '.', color="red", alpha=0.6)
            plt.plot(wlcs, catalog["mean_fluxerr"][order], '--', color="blue")
            plt.plot(wlcs, catalog["group_noise"][order], '-', color="orange")

            #plt.plot(catalog["vald_wlc"][order], catalog["wlc"][order])

            #plt.plot(catalog["vald_wlc"][order][filter], catalog["doppler_v"][order][filter], '-', color='green')
            #plt.plot(catalog["vald_wlc"][order][filter], catalog["rotinst_fwhm"][order][filter], '--', color='green')
            #plt.plot(catalog["vald_wlc"][order][~filter], catalog["doppler_v"][order][~filter], '-', color='red')
            #plt.plot(catalog["vald_wlc"][order][~filter], catalog["rotinst_fwhm"][order][~filter], '--', color='red')

        nlines_in_order = len(wlcs[filter])
        print("Order {0} has {1} lines included in mask.".format(order, nlines_in_order))
        total_number_of_lines += nlines_in_order
        # Build mask in format needed for CCF an
        dwl = wlcs[filter] * mask_width / (constants.c/1000.)
        mask_wl0 = wlcs[filter] - dwl/2.
        mask_wlf = wlcs[filter] + dwl/2.
        mask_weight = catalog['depth'][order][filter]

        for i in range(len(mask_weight)) :
            if mask_weight[i] not in outmask_weight :
                outmask_order = np.append(outmask_order, order)
                outmask_wl0 = np.append(outmask_wl0, mask_wl0[i])
                outmask_wlf = np.append(outmask_wlf, mask_wlf[i])
                outmask_weight = np.append(outmask_weight, mask_weight[i])
    if plot :
        #plt.ylabel(r"MEASURED $\lambda$ [nm]")
        plt.ylabel(r"Flux")
        #plt.ylabel(r"FWHM [km/s]")
        plt.xlabel(r"$\lambda_c$ [nm]")
        plt.show()


    print("Total of {0} lines included in output mask.".format(total_number_of_lines))

    # Save updated mask into file:
    if outputmask != "":
        if include_orders_in_mask :
            save_ccf_mask(outputmask, outmask_wl0, outmask_wlf, outmask_weight, orders=outmask_order)
        else :
            sortmask = np.argsort(outmask_wl0)
            save_ccf_mask(outputmask, outmask_wl0[sortmask], outmask_wlf[sortmask], outmask_weight[sortmask], orders=[])

    loc = {}
    loc["mask_width"] = mask_width
    loc["wl0"] = outmask_wl0
    loc["wlf"] = outmask_wlf
    loc["weight"] = outmask_weight

    return loc


def save_ccf_mask(output, wl0, wlf, weight, orders=[]) :
    outfile = open(output,"w+")

    if len(orders) :
        for i in range(len(wl0)) :
            outfile.write("{0:.0f} {1:.6f} {2:.6f} {3:.5f}\n".format(orders[i], wl0[i], wlf[i], weight[i]))
    else :
        for i in range(len(wl0)) :
            outfile.write("{0:.6f} {1:.6f} {2:.5f}\n".format(wl0[i],wlf[i],weight[i]))

    outfile.close()


# =============================================================================
# Define ccf used functions
# =============================================================================

def get_ccf_mask(loc, filename, lowlim=0., uplim=1.) :
    """
    Get the CCF mask

    :param p: parameter dictionary, ParamDict containing constants
        Must contain at least:
                ccf_mask: string, the name (and or location) of the CCF
                          mask file
                ic_w_mask_min: float, the weight of the CCF mask (if 1 force
                               all weights equal)
                ic_mask_width: float, the width of the template line
                               (if 0 use natural
                log_opt: string, log option, normally the program name

    :param loc: parameter dictionary, ParamDict containing data

    :param filename: string or None, the filename and location of the ccf mask
                     file, if None then file names is gotten from p['CCF_MASK']
                     
    :return loc: parameter dictionary, the updated parameter dictionary
            Adds/updates the following:
                ll_mask_d: numpy array (1D), the size of each pixel
                           (in wavelengths)
                ll_mask_ctr: numpy array (1D), the central point of each pixel
                             (in wavelengths)
                w_mask: numpy array (1D), the weight mask
    """
    #func_name = __NAME__ + '.get_ccf_mask()'
    # get constants from p
    mask_min = 0.
    mask_width = 1.0
    # speed of light in km/s
    c = constants.c / 1000.0
    # get table if not found raise error
    colnames = ['ll_mask_s', 'll_mask_e', 'w_mask']
    ccfmask = Table.read(filename, format='ascii')
    oldcols = ccfmask.colnames
    for c_it, col in enumerate(colnames):
        ccfmask[oldcols[c_it]].name = col

    # calculate the difference in mask_e and mask_s
    ll_mask_d = np.array(ccfmask['ll_mask_e']) - np.array(ccfmask['ll_mask_s'])
    ll_mask_ctr = np.array(ccfmask['ll_mask_s']) + ll_mask_d * 0.5
    # if mask_width > 0 ll_mask_d is multiplied by mask_width/c
    if mask_width > 0:
        ll_mask_d = mask_width * np.array(ccfmask['ll_mask_s']) / c
    # make w_mask an array
    w_mask = np.array(ccfmask['w_mask'])
    # use w_min to select on w_mask or keep all if w_mask_min >= 1
    if mask_min < 1.0:
        mask = w_mask > mask_min
        ll_mask_d = ll_mask_d[mask]
        ll_mask_ctr = ll_mask_ctr[mask]
        w_mask = w_mask[mask]
    # else set all w_mask to one (and use all lines in file)
    else:
        w_mask = np.ones(len(ll_mask_d))

    cutoff_filter = np.where(np.logical_and(w_mask > lowlim, w_mask <= uplim))
    # add to loc
    loc['LL_MASK_D'] = ll_mask_d[cutoff_filter]
    loc['LL_MASK_CTR'] = ll_mask_ctr[cutoff_filter]
    loc['W_MASK'] = w_mask[cutoff_filter]
    return loc
