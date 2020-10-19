# -*- coding: iso-8859-1 -*-
"""
    Created on September 28 2020
    
    Description:  library to handle SPIRou data
    
    @author: Eder Martioli <emartioli@lna.br>, <martioli@iap.fr>
    
    Laboratorio Nacional de Astrofisica, Brazil.
    
    Institut d'Astrophysique de Paris, France.
    
    """

__version__ = "1.0"

__copyright__ = """
    Copyright (c) ...  All rights reserved.
    """

import os,sys

import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from scipy import constants
from scipy.interpolate import interp1d
from scipy.optimize import curve_fit
from copy import copy, deepcopy
from astropy.io import ascii
from scipy.interpolate import UnivariateSpline
from scipy import stats
import time
import scipy.signal as signal
import scipy.interpolate as sint
import pylab as pl
import scipy as sp
import warnings

# Functon below defines SPIRou spectral orders, useful wavelength ranges and the NIR bands
def spirou_order_mask():
    order_mask = [[0, 967, 980, 'Y'],
             [1, 977, 994, 'Y'],
             [2, 989, 1008,'Y'],
             [3, 1000.1, 1020,'Y'],
             [4, 1018, 1035,'Y'],
             [5, 1027.2, 1050,'Y'],
             [6, 1042, 1065,'Y'],
             [7, 1055, 1078,'Y'],
             [8, 1071.5, 1096,'Y'],
             [9, 1084.5, 1112.8,'Y'],
             [10, 1098, 1126.5,'J'],
             [11, 1118, 1142,'J'],
             [12, 1135.5, 1162,'J'],
             [13, 1150, 1178,'J'],
             [14, 1168, 1198,'J'],
             [15, 1186, 1216,'J'],
             [16, 1204, 1235,'J'],
             [17, 1223, 1255,'J'],
             [18, 1243, 1275,'J'],
             [19, 1263, 1297,'J'],
             [20, 1284, 1321,'J'],
             [21, 1306, 1344,'J'],
             [22, 1327.5, 1367.5,'J'],
             [23, 1350.1, 1392,'J'],
             [24, 1374.3, 1415,'J'],
             [25, 1399.7, 1443.6,'H'],
             [26, 1426, 1470.9,'H'],
             [27, 1453.5, 1499,'H'],
             [28, 1482, 1528.6,'H'],
             [29, 1512, 1557.5,'H'],
             [30, 1544, 1591.1,'H'],
             [31, 1576.6, 1623,'H'],
             [32, 1608.5, 1658.9,'H'],
             [33, 1643.5, 1695,'H'],
             [34, 1679.8, 1733,'H'],
             [35, 1718, 1772,'H'],
             [36, 1758, 1813.5,'H'],
             [37, 1800.7, 1856.5,'H'],
             [38, 1843.9, 1902, 'H'],
             [39, 1890, 1949.5,'H'],
             [40, 1938.4, 1999.5, 'H'],
             [41, 1989.5, 2052, 'K'],
             [42, 2043, 2108, 'K'],
             [43, 2100, 2166, 'K'],
             [44, 2160, 2228,'K'],
             [45, 2223.5, 2293.6,'K'],
             [46, 2291, 2363,'K'],
             [47, 2362, 2436,'K'],
             [48, 2440, 2510,'K']]
             
    outorders, wl0, wlf, colors = [], [], [], []
    for order in order_mask:
        outorders.append(order[0])
        wl0.append(order[1])
        wlf.append(order[2])
        colors.append(order[3])
    
    loc = {}
    loc['orders'] = outorders
    loc['wl0'] = wl0
    loc['wlf'] = wlf
    loc['colors'] = colors
    return loc


#--- Load spirou RVs from v.fits file (which are the default products at CADC)
def load_rv_shifts_from_rdb(rdbfile, time_unit="RJD") :
    
    data = ascii.read(rdbfile,data_start=2)
    
    if time_unit == "RJD" :
        bjds = np.array(data['rjd'])
    elif time_unit == "MJD" :
        bjds = np.array(data['rjd']) + 2400000.5
    elif time_unit == "RJD" :
        bjds = np.array(data['rjd']) + 2400000.
    else:
        print("ERROR: time unit not supported")
        exit()

    ccfrvs = np.array(data["vrad"])
    ccfrvcs = np.array(data["vrad"])
    rvdrifts = np.full_like(ccfrvs,0.)
    
    loc = {}
    loc['BJD'] = np.array(bjds)
    loc['CCFRV'] = np.array(ccfrvs)
    loc['RVDRIFT'] = np.array(rvdrifts)
    loc['CCFRVC'] = np.array(ccfrvcs)

    return loc


def get_normalization_factor(flux_order, i_max, norm_window=50) :
    min_i = i_max - norm_window
    max_i = i_max + norm_window
    if min_i < 0 :
        min_i = 0
        max_i = min_i + 2 * norm_window
    if max_i >= len(flux_order) :
        max_i = len(flux_order) - 1

    # Calculate normalization factor as the median of flux within window around maximum signal
    normalization_factor = np.nanmedian(flux_order[min_i:max_i])

    return normalization_factor


#### Function to get chunk data #########
def get_chunk_data(spectrum, wl0, wlf, order, rv_overscan = 100.0, source_rv=0.0, apply_BERV=True, cal_fiber=False, normalize=1, nan_pos_filter=True, plot=False) :
    
    loc = {}
    loc['order'] = order

    loc['filename'] = spectrum['filename']
    loc['wl0'] = wl0
    loc['wlf'] = wlf

    wlc = (wl0 + wlf) / 2.
    
    # set overscan to avoid edge issues
     # in km/s
    loc['rv_overscan'] = rv_overscan

    #rv_overscan = 0.0 # in km/s
    dwl_1 = rv_overscan * wl0 / (constants.c / 1000.)
    dwl_2 = rv_overscan * wlf / (constants.c / 1000.)
    
    # get BERV from header
    if apply_BERV :
        BERV = spectrum['header1']['BERV']
    else :
        BERV = 0.

    # get DETECTOR GAIN and READ NOISE from header
    gain, rdnoise = spectrum['header0']['GAIN'], spectrum['header0']['RDNOISE']

    if cal_fiber :
        if nan_pos_filter :
            # mask NaN values
            nanmask = np.where(~np.isnan(spectrum['FluxAB'][order]))
            # mask negative and zero values
            negmask = np.where(spectrum['FluxAB'][order][nanmask] > 0)
            # set calibration fiber flux and wavelength vectors
            flux = spectrum['FluxC'][order][nanmask][negmask] / spectrum['BlazeC'][order][nanmask][negmask]
            wave = spectrum['WaveC'][order][nanmask][negmask]
            # calculate flux variance
            fluxerr = np.sqrt((spectrum['FluxAB'][order][nanmask][negmask] + (rdnoise * rdnoise / gain * gain) ) / spectrum['BlazeAB'][order][nanmask][negmask])
        else :
            # set calibration fiber flux and wavelength vectors
            flux = spectrum['FluxC'][order] / spectrum['BlazeC'][order]
            wave = spectrum['WaveC'][order]
            # calculate flux variance
            fluxerr = np.sqrt((spectrum['FluxAB'][order] + (rdnoise * rdnoise / gain * gain) ) / spectrum['BlazeAB'][order])

    else :
        if nan_pos_filter :
            # mask NaN values
            nanmask = np.where(~np.isnan(spectrum['FluxAB'][order]))
            # mask negative and zero values
            negmask = np.where(spectrum['FluxAB'][order][nanmask] > 0)
            # set science fiber flux and wavelength vectors
            flux = spectrum['FluxAB'][order][nanmask][negmask] / spectrum['BlazeAB'][order][nanmask][negmask]
            # apply BERV correction - Barycentric Earth Radial Velocity (BERV)
            wave = spectrum['WaveAB'][order][nanmask][negmask] * ( 1.0 + (BERV - source_rv) / (constants.c / 1000.) )
            # calculate flux variance
            fluxerr = np.sqrt(spectrum['FluxAB'][order][nanmask][negmask] + (rdnoise * rdnoise / gain * gain)) / spectrum['BlazeAB'][order][nanmask][negmask]
            if 'Recon' in spectrum.keys():
                recon = spectrum['Recon'][order][nanmask][negmask]
        else :
            # set science fiber flux and wavelength vectors
            flux = spectrum['FluxAB'][order] / spectrum['BlazeAB'][order]
            # apply BERV correction - Barycentric Earth Radial Velocity (BERV)
            wave = spectrum['WaveAB'][order] * ( 1.0 + (BERV - source_rv) / (constants.c / 1000.) )
            # calculate flux variance
            fluxerr = np.sqrt(spectrum['FluxAB'][order] + (rdnoise * rdnoise / gain * gain)) / spectrum['BlazeAB'][order]
            # get telluric absorption spectrum
            if 'Recon' in spectrum.keys():
                recon = spectrum['Recon'][order]
        
    # set wavelength masks
    wlmask = np.where(np.logical_and(wave > wl0 - dwl_1, wave < wlf + dwl_2))

    if len(flux[wlmask]) == 0 :
        loc['wl'] = np.array([])
        loc['flux'] = np.array([])
        loc['fluxerr'] = np.array([])
        if 'Recon' in spectrum.keys():
            loc['recon'] = np.array([])
        return loc

    # measure continuum and normalize flux if nomalize=True
    if normalize == 0:
        # Calculate normalization factor
        normalization_factor = spectrum['normalization_factor']

        loc['normalization_factor'] = normalization_factor

        # normalize flux
        flux = flux / normalization_factor
        fluxerr = fluxerr / normalization_factor

    elif normalize == 1:
        # Calculate normalization factor
        normalization_factor = get_normalization_factor(flux, np.nanargmax(spectrum['FluxAB'][order]))
        loc['normalization_factor'] = normalization_factor
        '''
         if plot :
            plt.plot(wave,flux,'.')
            plt.show()
         '''
        # normalize flux
        flux = flux / normalization_factor
        fluxerr = fluxerr / normalization_factor

    elif normalize == 2 :
        # get masked data
        flux_tmp = deepcopy(flux)
        wl_tmp = deepcopy(wave)
        loc['cont'] = np.array([])
        
        if len(wl_tmp[wlmask]) :
            '''
            # measure continuum
            cont, xbin, ybin = continuum(wl_tmp, flux_tmp, binsize=100,overlap=25, window=2,mode="max", use_linear_fit=True)
            if plot :
                plt.plot(wave,flux,'.')
                plt.plot(xbin,ybin, 'o')
                plt.plot(wave,cont, '-')
                plt.show()
            '''
            cont = fit_continuum(wl_tmp, flux_tmp, function='polynomial', order=4, nit=5, rej_low=2.0, rej_high=2.5, grow=1, med_filt=0, percentile_low=0., percentile_high=100., min_points=10, xlabel="wavelength (nm)", ylabel="flux", plot_fit=plot, verbose=False)
            
            loc['cont'] = cont[wlmask]
            # normalize flux
            flux = flux / cont
            fluxerr = fluxerr / cont

    
    # mask data
    flux, fluxerr, wave = flux[wlmask], fluxerr[wlmask], wave[wlmask]
    if 'Recon' in spectrum.keys():
        recon = recon[wlmask]

    if plot :
        plt.plot(wave,flux)
        plt.errorbar(wave,flux,yerr=fluxerr,fmt='.')
        if 'Recon' in spectrum.keys():
            plt.plot(wave,flux*recon,'-',linewidth=0.3)
        plt.show()

    loc['order'] = order
    loc['wl'] = wave
    if cal_fiber :
        loc['flux'] = flux / np.max(flux)
        loc['fluxerr'] = fluxerr / np.max(flux)
    else :
        loc['flux'] = flux
        loc['fluxerr'] = fluxerr
        if 'Recon' in spectrum.keys():
            loc['recon'] = recon


    return loc
##-- end of function


#--- Load a spirou spectrum from e.fits or t.fits file (which are the default products at CADC)
# This function preserves the spectral order structure
def load_spirou_AB_efits_spectrum(input, nan_pos_filter=True, preprocess=False, apply_BERV_to_preprocess=True, source_rv=0., normalization_in_preprocess=1, normalize_blaze=True) :
    
    # open fits file
    hdu = fits.open(input)
    
    if input.endswith("e.fits") :
        WaveAB = hdu["WaveAB"].data
        FluxAB = hdu["FluxAB"].data
        #BlazeAB = hdu[9].data / np.median(hdu[9].data)
        if normalize_blaze :
            BlazeAB = hdu["BlazeAB"].data / np.nanmean(hdu["BlazeAB"].data)
        else :
            BlazeAB = hdu["BlazeAB"].data
        
        WaveC = hdu["WaveC"].data
        FluxC = hdu["FluxC"].data
        #BlazeC = hdu["BlazeC"].data / np.median(hdu["BlazeC"].data)
        BlazeC = hdu["BlazeC"].data / np.nanmean(hdu["BlazeC"].data)

    elif input.endswith("t.fits") :
        WaveAB = hdu["WaveAB"].data
        FluxAB = hdu["FluxAB"].data
        #BlazeAB = hdu[3].data / np.median(hdu[3].data)
        if normalize_blaze :
            BlazeAB = hdu["BlazeAB"].data / np.nanmean(hdu["BlazeAB"].data)
        else :
            BlazeAB = hdu["BlazeAB"].data
        Recon = hdu["Recon"].data
    else :
        print("ERROR: unsupported extension for input file {}".format(input))
        exit()

    WaveABout, FluxABout, BlazeABout = [], [], []
    WaveCout, FluxCout, BlazeCout = [], [], []
    Reconout = []
    for i in range(len(WaveAB)) :
        if nan_pos_filter :
            # mask NaN values
            nanmask = np.where(~np.isnan(FluxAB[i]))
            # mask negative and zero values
            negmask = np.where(FluxAB[i][nanmask] > 0)

            WaveABout.append(WaveAB[i][nanmask][negmask])
            FluxABout.append(FluxAB[i][nanmask][negmask])
            BlazeABout.append(BlazeAB[i][nanmask][negmask])
            if input.endswith("e.fits") :
                WaveCout.append(WaveC[i][nanmask][negmask])
                FluxCout.append(FluxC[i][nanmask][negmask])
                BlazeCout.append(BlazeC[i][nanmask][negmask])
            elif input.endswith("t.fits") :
                Reconout.append(Recon[i][nanmask][negmask])
        else :
            WaveABout.append(WaveAB[i])
            FluxABout.append(FluxAB[i])
            BlazeABout.append(BlazeAB[i])
            if input.endswith("e.fits") :
                WaveCout.append(WaveC[i])
                FluxCout.append(FluxC[i])
                BlazeCout.append(BlazeC[i])
            elif input.endswith("t.fits") :
                Reconout.append(Recon[i])

    loc = {}
    loc['filename'] = input
    loc['header0'] = hdu[0].header
    loc['header1'] = hdu[1].header

    loc['WaveAB'] = WaveABout
    loc['FluxAB'] = FluxABout
    loc['BlazeAB'] = BlazeABout
    
    if input.endswith("e.fits") :
        loc['WaveC'] = WaveCout
        loc['FluxC'] = FluxCout
        loc['BlazeC'] = BlazeCout
        loc['headerC'] = hdu['FluxC'].header

    elif input.endswith("t.fits") :
        loc['Recon'] = Reconout

    if preprocess :
        # Pre-process spectrum to normalize data, remove nans and zeros, apply BERV correction if requested, etc
        loc = pre_process(loc, apply_BERV=apply_BERV_to_preprocess, source_rv=source_rv, normalize=normalization_in_preprocess, nan_pos_filter=nan_pos_filter)

    return loc


def pre_process(spectrum, apply_BERV=True, source_rv=0., normalize=1, nan_pos_filter=True) :
    orders = spirou_order_mask()
    out_wl, out_flux, out_fluxerr = [], [], []
    
    if normalize == 0:
        norm_chunk = get_chunk_data(spectrum, 1054., 1058., 6, rv_overscan = 0., source_rv=0.0, apply_BERV=False, cal_fiber=False, normalize=-1, nan_pos_filter=True, plot=False)
        spectrum["normalization_factor"] = np.median(norm_chunk['flux'])

    out_continuum = []
    
    for order in range(len(orders['orders'])) :
        wl0, wlf = orders['wl0'][order], orders['wlf'][order]
        
        try :
            loc = get_chunk_data(spectrum, wl0, wlf, order, rv_overscan=0., source_rv=source_rv, cal_fiber=False, apply_BERV=apply_BERV, normalize=normalize, nan_pos_filter=nan_pos_filter, plot=False)
            out_wl.append(loc['wl'])
            out_flux.append(loc['flux'])
            out_fluxerr.append(loc['fluxerr'])
            if normalize == 1 :
                out_continuum.append(loc['normalization_factor'])
            elif normalize == 2 :
                out_continuum.append(loc['cont'])
        except:
            out_wl.append(np.array([]))
            out_flux.append(np.array([]))
            out_fluxerr.append(np.array([]))
            if normalize == 1 or normalize == 2 :
                out_continuum.append(np.array([]))

    spectrum['wl'] = out_wl
    spectrum['flux'] = out_flux
    spectrum['fluxerr'] = out_fluxerr
    if normalize == 1 or normalize == 2:
        spectrum['normalization_factor'] = out_continuum

    return spectrum


########################################################################
def build_template(flux, wl, sig_clip = 0.0, fit=False, verbose=False, median=True, subtract=False, interpolate_nans=False):
    """
        Compute the median flux along the time axis
        Divide each exposure by the median
        
        Inputs:
        - flux: 2D matrix (N_exposures,N_wavelengths) from which median is computed
        - wl: 1D vector of floats with inoput wavelengths
        - fit: boolean to fit median spectrum to each observation before normalizing it
        Outputs:
        - loc: python dict containing all products
    """
    
    loc = {}

    if median :
        flux_med = np.median(flux,axis=0)
    else :
        flux_med = np.nanmean(flux,axis=0)

    if fit :
        shift_arr = []
        flux_calib = []
        flux_fit = []
        
        for i in range(len(flux)):
            guess = [0.0001]
            mask = ~np.isnan(flux[i])

            def flux_model (x, shift):
                outmodel = flux_med[mask] + shift
                return outmodel
                    
            if len(flux[i][mask]) > 0 :
                pfit, pcov = curve_fit(flux_model, wl[mask], flux[i][mask], p0=guess)
            else :
                pfit = [0.]
    
            shift_arr.append(pfit[0])
            flux_fit.append(flux_med + pfit[0])
            flux_calib.append(flux[i] - pfit[0])

        loc["shift"] = np.array(shift_arr, dtype=float)
        flux_calib = np.array(flux_calib, dtype=float)
        flux_fit = np.array(flux_fit, dtype=float)

        # Compute median on all spectra along the time axis
        if median :
            flux_med_new = np.median(flux_calib,axis=0)
        else :
            flux_med_new = np.nanmean(flux_calib,axis=0)

        flux_med = flux_med_new
        if subtract :
            flux_sub = flux_calib - flux_med
        else :
            flux_sub = flux_calib / flux_med

        residuals = flux_calib - flux_med
        flux_medsig = np.median(np.abs(residuals),axis=0) / 0.67449

    else :
        # Divide or subtract each flux by flux_med
        if subtract :
            flux_sub = flux - flux_med
        else :
            flux_sub = flux / flux_med

        residuals = flux - flux_med
        flux_medsig = np.median(np.abs(residuals),axis=0) / 0.67449

    # 1D quantities:
    # Fill in NaN's...
    if interpolate_nans :
        med_nan_mask = np.isnan(flux_med)
        flux_med[med_nan_mask] = np.interp(np.flatnonzero(med_nan_mask), np.flatnonzero(~med_nan_mask), flux_med[~med_nan_mask])
        flux_medsig[med_nan_mask] = np.interp(np.flatnonzero(med_nan_mask), np.flatnonzero(~med_nan_mask), flux_medsig[~med_nan_mask])
    loc["flux_med"] = flux_med
    loc["flux_sig"] = flux_medsig
    loc["wl"] = wl

    # 2D quantities:
    # Fill in residuals NaN's...
    if interpolate_nans :
        mask = np.isnan(residuals)
        residuals[mask] = np.interp(np.flatnonzero(mask), np.flatnonzero(~mask), residuals[~mask])
    loc["residuals"] = residuals
    # Fill in flux_sub NaN's...
    if interpolate_nans :
        masksub = np.isnan(flux_sub)
        flux_sub[masksub] = np.interp(np.flatnonzero(masksub), np.flatnonzero(~masksub), flux_sub[~masksub])

    loc["flux_sub"] = flux_sub
    loc["flux"] = loc["residuals"] + loc["flux_med"]

    return loc

#--- Function to create a template from a series of spirou spectra, applying the interpolation method
# this function keeps the spectral order structure
def template_using_fit(inputdata, rv_filename, median=False, normalize_by_continuum=False, verbose=False, plot=False, outputplot="") :
    
    warnings.simplefilter('ignore', np.RankWarning)
    
    # initialize loc dictionary to store data
    loc = {}
    
    # store input list of data files
    loc['inputdata'] = inputdata
    # store input list of rv data files
    loc['rv_filename'] = rv_filename

    #############################
    # Load bjd, airmass, SNR, and spectra data
    #############################
    bjd, airmass, snr = [], [], []
    berv = []
    spectra=[]
    # loop over all files in list
    for i in range(len(inputdata)) :
        if verbose:
            print("Loading spectrum {} -> {}/{}".format(inputdata[i],i,len(inputdata)-1))
        # load header extensions
        header0 = fits.getheader(inputdata[i],0)
        header1 = fits.getheader(inputdata[i],1)
        bjd.append((header0+header1)["BJD"])
        if "AIRMASS" in (header0+header1).keys() :
            airmass.append((header0+header1)["AIRMASS"])
        else :
            airmass.append(1.)

        if "SPEMSNR" in (header0+header1).keys() :
            snr.append((header0+header1)["SPEMSNR"])
        elif "SNR33" in (header0+header1).keys() :
            snr.append((header0+header1)["SNR33"])
        else :
            snr.append(1.)

        berv.append((header0+header1)["BERV"])
        # load SPIRou spectrum
        spectrum = load_spirou_AB_efits_spectrum(inputdata[i], nan_pos_filter=False, preprocess=True, apply_BERV_to_preprocess=False, source_rv=0., normalization_in_preprocess=0, normalize_blaze=True)
        spectra.append(spectrum)
    
    bjd = np.array(bjd, dtype=float)
    airmass = np.array(airmass, dtype=float)
    snr = np.array(snr, dtype=float)
    berv = np.array(berv, dtype=float)
    if rv_filename != "":
        # load vector of rv doppler shifts in the rv data file
        rv_loc = load_rv_shifts_from_rdb(rv_filename)
        rvs = rv_loc['CCFRV']
    else :
        rvs = np.zeros_like(berv)
    rvshifts = 1.0 + (berv - rvs)/(constants.c / 1000.)
    ###############################
    
    loc["bjd"] = bjd
    loc["airmass"] = airmass
    loc["snr"] = snr
    loc["berv"] = berv
    loc["berv_mean"] = np.mean(berv)
    loc["berv_sigma"] = np.std(berv)
    loc["rv_mean"] = np.mean(rvs)
    loc["rv_sigma"] = np.std(rvs)
    
    mean_rv_shift = 1.0 + (loc["berv_mean"] - loc["rv_mean"])/(constants.c / 1000.)
    max_rv_shift = 1.0 + np.max(np.abs(berv-rvs))/(constants.c / 1000.)
    min_rv_shift = 1.0 - np.max(np.abs(berv-rvs))/(constants.c / 1000.)

    if verbose :
        print("Mean BERV = {0:.3f}+-{1:.3f} km/s".format(loc["berv_mean"],loc["berv_sigma"]))
        print("Mean Source RV = {0:.3f}+-{1:.3f} km/s".format(loc["rv_mean"],loc["rv_sigma"]))

    # obtain information about spirou orders
    loc_orders = spirou_order_mask()

    # initialize output template vectors
    wl_template = []
    flux_template = []
    fluxerr_template = []
    
    for order in range(len(loc_orders['orders'])) :
    #for order in range(29,37) :
        if verbose:
            print("Processing order {}/{}".format(order,len(loc_orders['orders'])-1))

        # initialize global wl and flux vectors
        spectra_flux, spectra_fluxerr = [], []
        has_valid_data = False
        # loop over all files in list
        
        for i in range(len(inputdata)) :
            spectrum = spectra[i]
            if i == 0:
                # create output wavelength vector based on the vector of first spectrum and maximum
                # shifts caused by BERV and source RV.
                wl_tmp = spectrum['wl'][order] * mean_rv_shift

                # first calculate maximum velocity shift to cut edges to avoid interpolation issues
                min_out_wl = wl_tmp[0] * max_rv_shift
                max_out_wl = wl_tmp[-1] * min_rv_shift
                
                # create mask and output wl vector
                wlmask = wl_tmp > min_out_wl
                wlmask &= wl_tmp < max_out_wl
                spectra_wl = wl_tmp[wlmask]
            
            nanmask = ~np.isnan(spectrum['flux'][order])
            
            loc_wl = spectrum['wl'][order][nanmask] * rvshifts[i]
            loc_flux = spectrum['flux'][order][nanmask]
            loc_fluxerr = spectrum['fluxerr'][order][nanmask]
            
            if len(loc_flux) :
                has_valid_data = True
                
                out_flux = interp_spectrum(spectra_wl, loc_wl, loc_flux, kind='linear')
                out_fluxerr = interp_spectrum(spectra_wl, loc_wl, loc_fluxerr, kind='linear')
                
                spectra_flux.append(out_flux)
                spectra_fluxerr.append(out_fluxerr)
            else :
                spectra_flux.append(np.array([]))
                spectra_fluxerr.append(np.array([]))


        if has_valid_data :
            
            spectra_flux = np.array(spectra_flux, dtype=float)
            spectra_fluxerr = np.array(spectra_fluxerr, dtype=float)

            # calculate median spectra and residuals
            reduced_spectra = build_template(spectra_flux, spectra_wl, fit=True, median=median, subtract=True)
        
            # calculate median spectra using new calibrated fluxes-- 2nd pass
            reduced_spectra_2 = build_template(reduced_spectra["flux"], spectra_wl, fit=True, median=median, subtract=True)

            if plot :
                sig_clip = 3
                for i in range(len(reduced_spectra_2["flux"])) :
                    plt.plot(reduced_spectra_2["wl"], reduced_spectra_2["flux"][i], "-", color='#1f77b4', lw=0.6, alpha=0.3)
                    plt.plot(reduced_spectra_2["wl"], reduced_spectra_2["residuals"][i],"-", color='#1f77b4', lw=0.6, alpha=0.3)
                plt.plot(reduced_spectra_2["wl"],reduced_spectra["flux_med"],"-", color="red", lw=2)
                plt.plot(reduced_spectra_2["wl"], sig_clip * reduced_spectra["flux_sig"],"--", color="olive", lw=0.8)
                plt.plot(reduced_spectra_2["wl"],-sig_clip * reduced_spectra["flux_sig"],"--", color="olive", lw=0.8)

            if normalize_by_continuum and len(reduced_spectra_2['wl']):
                #print("Calculating continuum for normalization ...")
                cont = fit_continuum(reduced_spectra_2['wl'], reduced_spectra_2['flux_med'], function='polynomial', order=4, nit=5, rej_low=2.0, rej_high=2.5, grow=1, med_filt=0, percentile_low=0., percentile_high=100., min_points=10, xlabel="wavelength (nm)", ylabel="flux", plot_fit=False, verbose=False)
                reduced_spectra_2['flux_med'] /= cont
                reduced_spectra_2['flux_sig'] /= cont

                if plot :
                    plt.plot(reduced_spectra_2["wl"], reduced_spectra_2["flux_med"], "-", color="blue", lw=1, alpha=0.6)

            wl_template.append(reduced_spectra_2['wl'])
            flux_template.append(reduced_spectra_2['flux_med'])
            fluxerr_template.append(reduced_spectra_2['flux_sig'])
        else :
            wl_template.append(np.array([]))
            flux_template.append(np.array([]))
            fluxerr_template.append(np.array([]))

    if plot :
        plt.legend()
        plt.xlabel(r"$\lambda$ [nm]")
        plt.ylabel(r"Flux")
        if outputplot != "" :
            plt.savefig(outputplot, format='png')
        else :
            plt.show()
        #plt.clf()
        #plt.close()

    loc['wl'] = wl_template
    loc['flux'] = flux_template
    loc['fluxerr'] = fluxerr_template

    return loc


# function to interpolate spectrum
def interp_spectrum(wl_out, wl_in, flux_in, kind='cubic') :
    wl_in_copy = deepcopy(wl_in)
    
    # create interpolation function for input data
    f = interp1d(wl_in_copy, flux_in, kind=kind)
    
    # create mask for valid range of output vector
    mask = wl_out > wl_in[0]
    mask &= wl_out < wl_in[-1]
    
    flux_out = np.full_like(wl_out, np.nan)
    
    # interpolate data
    flux_out[mask] = f(wl_out[mask])
    return flux_out


#### Function to detect continuum #########
def continuum(x, y, binsize=200, overlap=100, sigmaclip=3.0, window=3,
              mode="median", use_linear_fit=False, telluric_bands=[], outx=None):
    """
    Function to calculate continuum
    :param x,y: numpy array (1D), input data (x and y must be of the same size)
    :param binsize: int, number of points in each bin
    :param overlap: int, number of points to overlap with adjacent bins
    :param sigmaclip: int, number of times sigma to cut-off points
    :param window: int, number of bins to use in local fit
    :param mode: string, set combine mode, where mode accepts "median", "mean",
                 "max"
    :param use_linear_fit: bool, whether to use the linar fit
    :param telluric_bands: list of float pairs, list of IR telluric bands, i.e,
                           a list of wavelength ranges ([wl0,wlf]) for telluric
                           absorption
    
    :return continuum, xbin, ybin
        continuum: numpy array (1D) of the same size as input arrays containing
                   the continuum data already interpolated to the same points
                   as input data.
        xbin,ybin: numpy arrays (1D) containing the bins used to interpolate
                   data for obtaining the continuum
    """

    if outx is None :
        outx = x
    
    # set number of bins given the input array length and the bin size
    nbins = int(np.floor(len(x) / binsize)) + 1

    # initialize arrays to store binned data
    xbin, ybin = [], []
    
    for i in range(nbins):
        # get first and last index within the bin
        idx0 = i * binsize - overlap
        idxf = (i + 1) * binsize + overlap
        # if it reaches the edges then reset indexes
        if idx0 < 0:
            idx0 = 0
        if idxf >= len(x):
            idxf = len(x) - 1
        # get data within the bin
        xbin_tmp = np.array(x[idx0:idxf])
        ybin_tmp = np.array(y[idx0:idxf])

        # create mask of telluric bands
        telluric_mask = np.full(np.shape(xbin_tmp), False, dtype=bool)
        for band in telluric_bands :
            telluric_mask += (xbin_tmp > band[0]) & (xbin_tmp < band[1])

        # mask data within telluric bands
        xtmp = xbin_tmp[~telluric_mask]
        ytmp = ybin_tmp[~telluric_mask]
        
        # create mask to get rid of NaNs
        nanmask = np.logical_not(np.isnan(ytmp))
        
        if i == 0 and not use_linear_fit:
            xbin.append(x[0] - np.abs(x[1] - x[0]))
            # create mask to get rid of NaNs
            localnanmask = np.logical_not(np.isnan(y))
            ybin.append(np.median(y[localnanmask][:binsize]))
        
        if len(xtmp[nanmask]) > 2 :
            # calculate mean x within the bin
            xmean = np.mean(xtmp[nanmask])
            # calculate median y within the bin
            medy = np.median(ytmp[nanmask])

            # calculate median deviation
            medydev = np.median(np.absolute(ytmp[nanmask] - medy))
            # create mask to filter data outside n*sigma range
            filtermask = (ytmp[nanmask] > medy) & (ytmp[nanmask] < medy +
                                                   sigmaclip * medydev)
            if len(ytmp[nanmask][filtermask]) > 2:
                # save mean x wihthin bin
                xbin.append(xmean)
                if mode == 'max':
                    # save maximum y of filtered data
                    ybin.append(np.max(ytmp[nanmask][filtermask]))
                elif mode == 'median':
                    # save median y of filtered data
                    ybin.append(np.median(ytmp[nanmask][filtermask]))
                elif mode == 'mean':
                    # save mean y of filtered data
                    ybin.append(np.mean(ytmp[nanmask][filtermask]))
                else:
                    emsg = 'Can not recognize selected mode="{0}"...exiting'
                    print('error', emsg.format(mode))

        if i == nbins - 1 and not use_linear_fit:
            xbin.append(x[-1] + np.abs(x[-1] - x[-2]))
            # create mask to get rid of NaNs
            localnanmask = np.logical_not(np.isnan(y[-binsize:]))
            ybin.append(np.median(y[-binsize:][localnanmask]))

    # Option to use a linearfit within a given window
    if use_linear_fit:
        # initialize arrays to store new bin data
        newxbin, newybin = [], []

        # loop around bins to obtain a linear fit within a given window size
        for i in range(len(xbin)):
            # set first and last index to select bins within window
            idx0 = i - window
            idxf = i + 1 + window
            # make sure it doesnt go over the edges
            if idx0 < 0: idx0 = 0
            if idxf > nbins: idxf = nbins - 1

            # perform linear fit to these data
            slope, intercept, r_value, p_value, std_err = stats.linregress(xbin[idx0:idxf], ybin[idx0:idxf])

            if i == 0 :
                # append first point to avoid crazy behaviours in the edge
                newxbin.append(x[0] - np.abs(x[1] - x[0]))
                newybin.append(intercept + slope * newxbin[0])
            
            # save data obtained from the fit
            newxbin.append(xbin[i])
            newybin.append(intercept + slope * xbin[i])

            if i == len(xbin) - 1 :
                # save data obtained from the fit
                newxbin.append(x[-1] + np.abs(x[-1] - x[-2]))
                newybin.append(intercept + slope * newxbin[-1])

        xbin, ybin = newxbin, newybin

    # interpolate points applying an Spline to the bin data
    sfit = UnivariateSpline(xbin, ybin, s=0)
    #sfit.set_smoothing_factor(0.5)
    
    # Resample interpolation to the original grid
    cont = sfit(outx)

    # return continuum and x and y bins
    return cont, xbin, ybin
##-- end of continuum function


def write_spectrum_orders_to_fits(spectrum, filename, header=None, wavekey='wl', fluxkey='flux', fluxerrkey='fluxerr'):

    if header is None :
        header = fits.Header()
    header.set('ORIGIN', "spirou-tools")
    header.set('UTCSAVED', time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime()))

    maxlen = 0
    for order in range(len(spectrum[wavekey])) :
        if len(spectrum[wavekey][order]) > maxlen :
            maxlen = len(spectrum[wavekey][order])

    wl_data = np.full((len(spectrum[wavekey]),maxlen), np.nan)
    flux_data = np.full((len(spectrum[wavekey]),maxlen), np.nan)
    err_data = np.full((len(spectrum[wavekey]),maxlen), np.nan)

    for order in range(len(spectrum[wavekey])) :
        for i in range(len(spectrum[wavekey][order])) :
            wl_data[order][i] = spectrum[wavekey][order][i]
            flux_data[order][i] = spectrum[fluxkey][order][i]
            if fluxerrkey in spectrum.keys():
                err_data[order][i] = spectrum[fluxerrkey][order][i]

    header.set('TTYPE1', "WAVE")
    header.set('TUNIT1', "NM")
    header.set('TTYPE2', "FLUX")
    header.set('TUNIT2', "COUNTS")

    if fluxerrkey in spectrum.keys():
        header.set('TTYPE2', "FLUXERR")
        header.set('TUNIT2', "COUNTS")

    primary_hdu = fits.PrimaryHDU(header=header)
    hdu_wl = fits.ImageHDU(data=wl_data, name="WAVE")
    hdu_flux = fits.ImageHDU(data=flux_data, name="FLUX")
    if fluxerrkey in spectrum.keys():
        hdu_err = fits.ImageHDU(data=err_data, name="FLUXERR")
        mef_hdu = fits.HDUList([primary_hdu, hdu_wl, hdu_flux, hdu_err])
    else :
        mef_hdu = fits.HDUList([primary_hdu, hdu_wl, hdu_flux])

    mef_hdu.writeto(filename, overwrite=True)


def read_orders_spectrum_from_fits(filename) :
    hdu = fits.open(filename)
    
    spectrum = {}
    spectrum['header'] = hdu[0].header
    
    wave = hdu['WAVE'].data
    flux = hdu['FLUX'].data
    fluxerr = hdu['FLUXERR'].data

    wl_out, flux_out, fluxerr_out = [], [], []

    for order in range(len(wave)):
        nanmask = np.where(np.logical_and(flux[order] > 0, ~np.isnan(flux[order])))
        wl_out.append(wave[order][nanmask])
        flux_out.append(flux[order][nanmask])
        fluxerr_out.append(fluxerr[order][nanmask])

    spectrum['wl'] = wl_out
    spectrum['flux'] = flux_out
    spectrum['fluxerr'] = fluxerr_out
    
    return spectrum


def write_spectrum_to_fits(spectrum, filename, header=None, wavekey='wl', fluxkey='flux', fluxerrkey='fluxerr'):
    
    '''
        Function to save input spectrum as FITS file (Adapted from write_spectrum function of iSpec)
        Input spectrum is a dictionary with the following data:
        - wavelength (nm): spectrum[wavekey],
        - flux: spectrum[fluxkey],
        - flux error: spectrum[fluxerrkey] (optional)
        If header is input then it will adds variables to the existing header and save to the output fits
     '''
    
    if type(spectrum[wavekey]) == list :
        write_spectrum_orders_to_fits(spectrum, filename, header=header, wavekey=wavekey, fluxkey=fluxkey, fluxerrkey=fluxerrkey)
        return
    
    if header is None :
        header = fits.Header()

    header.set('ORIGIN', "spirou-tools")
    header.set('UTCSAVED', time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime()))

    wave_diff = spectrum[wavekey][1:] - spectrum[wavekey][:-1]
    median_wave_step = np.median(wave_diff)
    if np.all(np.abs(wave_diff - median_wave_step) < 0.0000001):
        ### Regularly sampled spectrum
        primary_data = np.asarray(spectrum[fluxkey], dtype='float32')

        # Coordinates
        header.set('CUNIT1', "NM")
        header.set('CTYPE1', "WAVE") # wavelength
        #header.set('CD1_1', spectrum[wavekey][1] - spectrum[wavekey][0])
        header.set('CDELT1', spectrum[wavekey][1] - spectrum[wavekey][0])
        header.set('CRVAL1', spectrum[wavekey][0])
        header.set('CRPIX1', 1)
        #
        header.set('NAXIS', 1)
        header.set('NAXIS1', len(spectrum[fluxkey]))
        header.set('TTYPE1', "FLUX")
        header.set('TUNIT1', "COUNTS")

        primary_hdu = fits.PrimaryHDU(data=primary_data, header=header)

        # Add an HDU extension (image) with errors if they exist
        if fluxerrkey in spectrum.keys():
            # Error extension
            ext_data = np.asarray(spectrum[fluxerrkey], dtype='float32')
            extheader = header.copy()
            extheader.set('TTYPE2', "FLUXERR")
            extheader.set('TUNIT2', "COUNTS")
            extheader.set('NAXIS', 1)
            extheader.set('NAXIS1', len(spectrum[fluxerrkey]))
            extension_hdu = fits.ImageHDU(data=ext_data, header=extheader, name="FLUXERR")
            fits_format = fits.HDUList([primary_hdu, extension_hdu])
        else:
            fits_format = fits.HDUList(primary_hdu)
    else:
        primary_hdu = fits.PrimaryHDU(header=header)
        cols = []
        cols.append(fits.Column(name='WAVE', format='E', array=spectrum[wavekey]))
        cols.append(fits.Column(name='FLUX', format='E', array=spectrum[fluxkey]))
        if fluxerrkey in spectrum.keys():
            cols.append(fits.Column(name='FLUXERR', format='E', array=spectrum[fluxerrkey]))
        
        hdu = fits.BinTableHDU.from_columns(cols)
        fits_format = fits.HDUList([primary_hdu, hdu])
    
    fits_format.writeto(filename, overwrite=True)


def fit_continuum(wav, spec, function='polynomial', order=3, nit=5, rej_low=2.0,
    rej_high=2.5, grow=1, med_filt=0, percentile_low=0., percentile_high=100.,
                  min_points=10, xlabel="", ylabel="", plot_fit=True, verbose=False):
    """
    Continuum fitting re-implemented from IRAF's 'continuum' function
    in non-interactive mode only but with additional options.

    :Parameters:
    
    wav: array(float)
        abscissa values (wavelengths, velocities, ...)

    spec: array(float)
        spectrum values

    function: str
        function to fit to the continuum among 'polynomial', 'spline3'

    order: int
        fit function order:
        'polynomial': degree (not number of parameters as in IRAF)
        'spline3': number of knots

    nit: int
        number of iteractions of non-continuum points
        see also 'min_points' parameter

    rej_low: float
        rejection threshold in unit of residul standard deviation for point
        below the continuum

    rej_high: float
        same as rej_low for point above the continuum

    grow: int
        number of neighboring points to reject

    med_filt: int
        median filter the spectrum on 'med_filt' pixels prior to fit
        improvement over IRAF function
        'med_filt' must be an odd integer

    percentile_low: float
        reject point below below 'percentile_low' percentile prior to fit
        improvement over IRAF function
        "percentile_low' must be a float between 0. and 100.

    percentile_high: float
        same as percentile_low but reject points in percentile above
        'percentile_high'
        
    min_points: int
        stop rejection iterations when the number of points to fit is less than
        'min_points'

    plot_fit: bool
        if true display two plots:
            1. spectrum, fit function, rejected points
            2. residual, rejected points

    verbose: bool
        if true fit information is printed on STDOUT:
            * number of fit points
            * RMS residual
    """
    mspec = np.ma.masked_array(spec, mask=np.zeros_like(spec))
    # mask 1st and last point: avoid error when no point is masked
    # [not in IRAF]
    mspec.mask[0] = True
    mspec.mask[-1] = True
    
    mspec = np.ma.masked_where(np.isnan(spec), mspec)
    
    # apply median filtering prior to fit
    # [opt] [not in IRAF]
    if int(med_filt):
        fspec = signal.medfilt(spec, kernel_size=med_filt)
    else:
        fspec = spec
    # consider only a fraction of the points within percentile range
    # [opt] [not in IRAF]
    mspec = np.ma.masked_where(fspec < np.percentile(fspec, percentile_low),
        mspec)
    mspec = np.ma.masked_where(fspec > np.percentile(fspec, percentile_high),
        mspec)
    # perform 1st fit
    if function == 'polynomial':
        coeff = np.polyfit(wav[~mspec.mask], spec[~mspec.mask], order)
        cont = np.poly1d(coeff)(wav)
    elif function == 'spline3':
        knots = wav[0] + np.arange(order+1)[1:]*((wav[-1]-wav[0])/(order+1))
        spl = sint.splrep(wav[~mspec.mask], spec[~mspec.mask], k=3, t=knots)
        cont = sint.splev(wav, spl)
    else:
        raise(AttributeError)
    # iteration loop: reject outliers and fit again
    if nit > 0:
        for it in range(nit):
            res = fspec-cont
            sigm = np.std(res[~mspec.mask])
            # mask outliers
            mspec1 = np.ma.masked_where(res < -rej_low*sigm, mspec)
            mspec1 = np.ma.masked_where(res > rej_high*sigm, mspec1)
            # exlude neighbors cf IRAF's continuum parameter 'grow'
            if grow > 0:
                for sl in np.ma.clump_masked(mspec1):
                    for ii in range(sl.start-grow, sl.start):
                        if ii >= 0:
                            mspec1.mask[ii] = True
                    for ii in range(sl.stop+1, sl.stop+grow+1):
                        if ii < len(mspec1):
                            mspec1.mask[ii] = True
            # stop rejection process when min_points is reached
            # [opt] [not in IRAF]
            if np.ma.count(mspec1) < min_points:
                if verbose:
                    print("  min_points %d reached" % min_points)
                break
            mspec = mspec1
            if function == 'polynomial':
                coeff = np.polyfit(wav[~mspec.mask], spec[~mspec.mask], order)
                cont = np.poly1d(coeff)(wav)
            elif function == 'spline3':
                knots = wav[0] + np.arange(order+1)[1:]*((wav[-1]-wav[0])/(order+1))
                spl = sint.splrep(wav[~mspec.mask], spec[~mspec.mask], k=3, t=knots)
                cont = sint.splev(wav, spl)
            else:
                raise(AttributeError)
    # compute residual and rms
    res = fspec-cont
    sigm = np.std(res[~mspec.mask])
    if verbose:
        print("  nfit=%d/%d" %  (np.ma.count(mspec), len(mspec)))
        print("  fit rms=%.3e" %  sigm)
    # compute residual and rms between original spectrum and model
    # different from above when median filtering is applied
    ores = spec-cont
    osigm = np.std(ores[~mspec.mask])
    if int(med_filt) and verbose:
        print("  unfiltered rms=%.3e" %  osigm)
    # plot fit results
    if plot_fit:
        # overplot spectrum and model + mark rejected points
        fig1 = pl.figure(1)
        ax1 = fig1.add_subplot(111)
        ax1.plot(wav[~mspec.mask], spec[~mspec.mask],
            c='tab:blue', lw=1.0)
        # overplot median filtered spectrum
        if int(med_filt):
            ax1.plot(wav[~mspec.mask], fspec[~mspec.mask],
                c='tab:cyan', lw=1.0)
        ax1.scatter(wav[mspec.mask], spec[mspec.mask], s=20., marker='d',
        edgecolors='tab:gray', facecolors='none', lw=0.5)
        ax1.plot(wav, cont, ls='--', c='tab:orange')
        if nit > 0:
            # plot residuals and rejection thresholds
            fig2 = pl.figure(2)
            ax2 = fig2.add_subplot(111)
            ax2.axhline(0., ls='--', c='tab:orange', lw=1.)
            ax2.axhline(-rej_low*sigm, ls=':')
            ax2.axhline(rej_high*sigm, ls=':')
            ax2.scatter(wav[mspec.mask], res[mspec.mask],
                s=20., marker='d', edgecolors='tab:gray', facecolors='none',
                lw=0.5)
            ax2.scatter(wav[~mspec.mask], ores[~mspec.mask],
                marker='o', s=10., edgecolors='tab:blue', facecolors='none',
                lw=.5)
            # overplot median filtered spectrum
            if int(med_filt):
                ax2.scatter(wav[~mspec.mask], res[~mspec.mask],
                    marker='s', s=5., edgecolors='tab:cyan', facecolors='none',
                    lw=.2)
        if xlabel != "" :
            pl.xlabel(xlabel)
        if ylabel != "" :
            pl.ylabel(ylabel)
        pl.show()
    return cont


def save_rv_time_series(output, bjd, rv, rverr, time_in_rjd=True, rv_in_mps=False) :
    
    outfile = open(output,"w+")
    outfile.write("rjd\tvrad\tsvrad\n")
    outfile.write("---\t----\t-----\n")
    
    for i in range(len(bjd)) :
        
        if time_in_rjd :
            rjd = bjd[i] - 2400000.
        else :
            rjd = bjd[i]
        
        if rv_in_mps :
            outfile.write("{0:.10f}\t{1:.2f}\t{2:.2f}\n".format(rjd, 1000. * rv[i], 1000. * rverr[i]))
        else :
            outfile.write("{0:.10f}\t{1:.5f}\t{2:.5f}\n".format(rjd, rv[i], rverr[i]))

    outfile.close()
