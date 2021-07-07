# -*- coding: iso-8859-1 -*-
"""
    Created on May 27 2021
    
    Description: utilities for calculation of instrumental drifts in SPIRou RV data
    
    @authors:  Eder Martioli <martioli@iap.fr>
    
    Institut d'Astrophysique de Paris, France.
    
    """

__version__ = "1.0"

__copyright__ = """
    Copyright (c) ...  All rights reserved.
    """
import os,sys
import numpy as np
import matplotlib.pyplot as plt

from astropy.io import ascii
import astropy.io.fits as fits
from scipy.optimize import curve_fit
from scipy import ndimage
from copy import deepcopy

from scipy.optimize import minimize

def get_rv_drifts(inputdata, drift_repository, correct_drift=True, use_linear_drift=True, verbose=False) :
    
    if correct_drift :
        jds, obsdates = [], []
    
        for exp in inputdata :
            hdr = fits.getheader(exp)
        
            obsdates.append(hdr['DATE-OBS'])
            jd = hdr['MJDATE'] + 2400000.5
            jds.append(jd)
    
        jds = np.array(jds)
        obsdates = np.array(obsdates)
        
        if verbose:
            print("Calculating RV drifts between wave sol and sim. FP CCF ... ")
        drifts, drifts_err = calculate_ccf_drifts(drift_repository, jds, obsdates, use_linear_drift=use_linear_drift, plot=True, verbose=verbose)
    else :
        drifts, drifts_err = np.zeros(len(inputdata)), np.zeros(len(inputdata))
    
    output = []
    
    for i in range(len(inputdata)) :
        loc = {}
        loc["WFPDRIFT"] = 'None' # Wavelength sol absolute CCF FP Drift [km/s]
        loc["RV_WAVFP"] = 'None' # RV measured from wave sol FP CCF [km/s]
        loc["RV_SIMFP"] = 'None' # RV measured from simultaneous FP CCF [km/s]
        loc["RV_DRIFT"] = drifts[i] # RV drift between wave sol and sim. FP CCF [km/s]
        loc["RV_DRIFTERR"] = drifts_err[i] # RV drift error between wave sol and sim. FP CCF [km/s]
        output.append(loc)
    
    return output


def calculate_ccf_drifts(drift_repository, jds, obsdates, use_linear_drift=True, plot=False, verbose=False) :

    drifts, drift_err = np.array([]), np.array([])
    
    nights = get_nights_observed(obsdates)

    for night in nights :
        same_night = obsdates == night
    
        drift_file = get_drift_file(drift_repository, night)
    
        if verbose :
            print("Calculating drifts for night: {} using drift file:{} ".format(night,drift_file))
        night_jds = deepcopy(jds[same_night])
        
        locdrifts, locdrift_err = calculate_drifts(drift_file, night_jds, night=night, median_filter=True, use_linear_drift=use_linear_drift, plot=plot, verbose=verbose)

        drifts = np.append(drifts, locdrifts)
        drift_err = np.append(drift_err, locdrift_err)

    return drifts, drift_err


def calculate_drifts(filename, jds, night='', median_filter=False, use_linear_drift=True, plot=False, verbose=False) :
    
    if not os.path.exists(filename) :
        print("WARNING: inexistent drift file: {} -> setting all drifts to zero".format(filename))
        return np.zeros_like(jds), np.zeros_like(jds)
    
    rvdata = ascii.read(filename, data_start=2)

    jd_data = np.array(rvdata['rjd']) + 2400000.

    if len(jd_data) < 2:
        print("WARNING: less than 2 points in drift file: {} -> setting all drifts to zero".format(filename))
        return np.zeros_like(jds), np.zeros_like(jds)

    vrad = np.array(rvdata['vrad'])
    svrad = np.array(rvdata['svrad'])

    non_finite_errors = ~np.isfinite(svrad)
    svrad[non_finite_errors] = 0.005

    densfilt = np.array(rvdata['SBCDEN_P'])
    dateobs = np.array(rvdata['DATE-OBS'])
    exptype = np.array(rvdata['EXPTYPE'])
    
    df_values = []
    for i in range(len(densfilt)) :
        if densfilt[i] not in df_values :
            df_values.append(densfilt[i])
    df_values.sort()

    # select calibration data (usually 2 points: afternoon and morning
    calib = exptype == 'ALIGN'

    ### first calculate linear drift in calibration points:
    drift_slope, drift_intercept = 0.,0.

    if len(jd_data[calib]) < 2:
        print("WARNING: less than 2 ALIGN points in drift file: {} -> setting all drifts to zero".format(filename))
        return np.zeros_like(jds), np.zeros_like(jds)

    elif len(jd_data[calib]) == 2:
        
        from uncertainties import ufloat
        x0, x1 = jd_data[calib][0], jd_data[calib][1]
        y0, y1 = ufloat(vrad[calib][0],svrad[calib][0]), ufloat(vrad[calib][1],svrad[calib][1])
        udrift_slope = (y1 - y0) / (x1 - x0)
        udrift_intercept0 = y0 - udrift_slope * x0
        udrift_intercept1 = y1 - udrift_slope * x1
        udrift_intercept = (udrift_intercept0 + udrift_intercept1) / 2

        drift_slope, drift_intercept = udrift_slope.nominal_value, udrift_intercept.nominal_value
    else :
        def linearfunc(t, a, b) :
            return a + b * t

        guess = [np.nanmean(vrad[calib]),0.000001]
        linfit, lincov = curve_fit(linearfunc, jd_data[calib], vrad[calib], p0=guess, sigma=svrad[calib])
        drift_slope, drift_intercept = linfit[1], linfit[0]

    rv0 = jd_data[calib][0] * drift_slope + drift_intercept
    linear_drift = jd_data * drift_slope + drift_intercept
    meanfprv = np.nanmean(linear_drift)
    linear_drift -= rv0
    
    if plot :
        #plt.errorbar(jd_data[calib], vrad[calib], yerr=svrad[calib], fmt='o')
        plt.plot(jd_data, linear_drift,"--",label='linear drift')
    #plt.show()

    cbjds, crvs, crverrs = np.array([]), np.array([]), np.array([])

    for i in range(len(df_values)) :
        df = df_values[i]
        color = [i/len(df_values),1-i/len(df_values),1-i/len(df_values)]
        dfmask = densfilt == df
        #plt.plot(jd_data[dfmask],densfilt[dfmask],'.')
        #plt.plot(jd_data[dfmask],vrad[dfmask],'o', color=color, alpha=0.35, label="D={0:.2f}".format(df))
        
        def rv_func(t, shift) :
            return t * drift_slope + drift_intercept + shift - rv0
        
        # Fit all data to match first GP model
        guess = [(meanfprv  - np.nanmedian(vrad[dfmask])) - rv0]
        pfit, pcov = curve_fit(rv_func, jd_data[dfmask], vrad[dfmask], p0=guess, sigma=svrad[dfmask])
        
        #print(i,df,meanfprv,np.nanmedian(vrad[dfmask]),pfit[0])
        fprv_calib = vrad[dfmask] - pfit[0]
        
        if plot :
            plt.errorbar(jd_data[dfmask], fprv_calib, yerr=svrad[dfmask], fmt='o', color=color, alpha=0.35, label="D={0:.2f}".format(df))

        cbjds = np.append(cbjds, jd_data[dfmask])
        crvs = np.append(crvs, fprv_calib)
        crverrs = np.append(crverrs, svrad[dfmask])

    sortInd = np.argsort(cbjds)
    cbjds, crvs, crverrs = cbjds[sortInd], crvs[sortInd], crverrs[sortInd]

    if use_linear_drift or len(cbjds) < 3:
        if verbose :
            print("Estimated linear drift: {} x JD + {} km/s".format(drift_slope,drift_intercept))
        preddrifts = jds * drift_slope + drift_intercept - rv0
        predstd = np.full_like(jds, np.nanmean(crverrs))
    else :

        import celerite
        from celerite import terms
        from celerite.modeling import Model

        # Define the model
        class MeanModel(Model):
            parameter_names = ("alpha", "ell", "log_sigma2")
    
            def get_value(self, t):
                return self.alpha * np.exp(-0.5*(t-self.ell)**2 * np.exp(-self.log_sigma2))
    
            # This method is optional but it can be used to compute the gradient of the
            # cost function below.
            def compute_gradient(self, t):
                e = 0.5*(t-self.ell)**2 * np.exp(-self.log_sigma2)
                dalpha = np.exp(-e)
                dell = self.alpha * dalpha * (t-self.ell) * np.exp(-self.log_sigma2)
                dlog_s2 = self.alpha * dalpha * e
                return np.array([dalpha, dell, dlog_s2])

        # Define a cost function
        def neg_log_like(params, y, gp):
            gp.set_parameter_vector(params)
            return -gp.log_likelihood(y)

        def grad_neg_log_like(params, y, gp):
            gp.set_parameter_vector(params)
            return -gp.grad_log_likelihood(y)[1]

        def gp_calib_fit_model(cbjds, crvs, crverrs) :
            # Set up the GP model
            kernel = terms.RealTerm(log_a=np.log(np.var(crvs)), log_c=-np.log(10.0))
            mean_model = MeanModel(alpha=-1.0, ell=0.1, log_sigma2=np.log(0.4))
            gp = celerite.GP(kernel, mean=mean_model, fit_mean=True)
            gp.compute(cbjds, crverrs)
            #   print("Initial log-likelihood: {0}".format(gp.log_likelihood(crvs)))
    
            # Fit for the maximum likelihood parameters
            initial_params = gp.get_parameter_vector()
            bounds = gp.get_parameter_bounds()
            soln = minimize(neg_log_like, initial_params, jac=grad_neg_log_like, method="L-BFGS-B",bounds=bounds, args=(crvs, gp))
            gp.set_parameter_vector(soln.x)
            #print("Final log-likelihood: {0}".format(-soln.fun))
            return gp

        if median_filter:
            med_filter_size = int(np.floor(len(cbjds)/10))
            crvs = ndimage.median_filter(crvs, size=med_filter_size)
            crverrs /= np.sqrt(med_filter_size)
            if plot :
                plt.plot(cbjds, crvs, 'k+', label="median filtered")

        # generate a first calibration model using GP
        gp1 = gp_calib_fit_model(cbjds, crvs, crverrs)

        mu, var = gp1.predict(crvs, cbjds, return_var=True)
        std = np.sqrt(var)

        if verbose :
            print("RMS of 1st calibration residuals: {0:.1f} m/s".format(1000.*np.nanstd(crvs - mu)))

        if plot :
            # Plot the data
            color = "#ff7f0e"
            plt.plot(cbjds, mu, color=color, label="GP prediction")
            plt.fill_between(cbjds, mu+std, mu-std, color=color, alpha=0.3, edgecolor="none")
            plt.gca().yaxis.set_major_locator(plt.MaxNLocator(5))

        preddrifts, predvar = gp1.predict(crvs, jds, return_var=True)
        predstd = np.sqrt(predvar)


    if plot :
        plt.errorbar(jds, preddrifts, yerr=predstd, fmt="ro",label="predicted drifts")
        if night != "":
            plt.title("{}".format(night))
        plt.ylabel("Drift velocity of simultaneous FP [km/s]")
        plt.xlabel("Julian Date")
        plt.legend(fontsize=5, ncol=4)
        plt.show()

    return preddrifts, predstd


def get_nights_observed(allobsdates) :
    obsdates = []
    for obsdate in allobsdates :
        if obsdate not in obsdates :
            obsdates.append(obsdate)
    return obsdates


def get_drift_file(drift_repository, night) :
    filename = "{0}/{1}_fiber_C_ccf_rv.rdb".format(drift_repository,night)
    return filename
