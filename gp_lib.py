# -*- coding: utf-8 -*-


# -*- coding: iso-8859-1 -*-
"""
    Created on Jun 19 2021
    
    Description: library with several utilities using Gaussian Process
    
    @authors:  Eder Martioli <martioli@iap.fr>
    
    Institut d'Astrophysique de Paris, France.
    
    """

__version__ = "1.0"

__copyright__ = """
    Copyright (c) ...  All rights reserved.
    """

import numpy as np
import matplotlib.pyplot as plt
from celerite.modeling import Model
from scipy.optimize import minimize
import celerite
from celerite import terms
import emcee
import corner


# Define the model
class gaussianModel(Model):
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


def fit_gaussian_profile(x, y, yerr, burnin=250, nsamples=1000, plot=False, verbose=False) :

    mean_model = gaussianModel(alpha=-1., ell=x[np.nanargmin(y)], log_sigma2=1.0)
    
    # Set up the GP model
    kernel = terms.RealTerm(log_a=np.log(np.var(y)), log_c=-np.log(10.0))
    gp = celerite.GP(kernel, mean=mean_model, fit_mean=True)
    
    gp.compute(x, yerr)
    if verbose :
        print("Initial log-likelihood: {0}".format(gp.log_likelihood(y)))

    # Fit for the maximum likelihood parameters
    initial_params = gp.get_parameter_vector()

    bounds = gp.get_parameter_bounds()

    soln = minimize(neg_log_like, initial_params, jac=grad_neg_log_like, method="L-BFGS-B", bounds=bounds, args=(y, gp))

    gp.set_parameter_vector(soln.x)
    if verbose :
        print("Final log-likelihood: {0}".format(-soln.fun))

    def log_probability(params):
        gp.set_parameter_vector(params)
        lp = gp.log_prior()
        if not np.isfinite(lp):
            return -np.inf
        return gp.log_likelihood(y) + lp
            
    initial = np.array(soln.x)
    ndim, nwalkers = len(initial), 32
    sampler = emcee.EnsembleSampler(nwalkers, ndim, log_probability)
    if verbose :
        print("Running burn-in...")
    p0 = initial + 1e-8 * np.random.randn(nwalkers, ndim)
    p0, lp, _ = sampler.run_mcmc(p0, burnin)

    print("Running production...")
    sampler.reset()
    sampler.run_mcmc(p0, nsamples, progress=True);
    samples = sampler.chain.reshape((-1, ndim))

    bestfit_params, bestfit_params_err = best_fit_parameters(samples)
    
    names = gp.get_parameter_names()
    
    for i in range(len(bestfit_params)) :
        print("{}={:.5f}+-{:.5f}".format(names[i],bestfit_params[i],bestfit_params_err[i]))
    
    gp.set_parameter_vector(bestfit_params)
    
    if plot :
        # get model
        xm = np.linspace(x[0], x[-1], 500)
        mu, var = gp.predict(y, xm, return_var=True)
        ym = mean_model.get_value(xm)
        std = np.sqrt(var)

        color = "#ff7f0e"

        # Plot 24 posterior samples.
        m_samples = sampler.flatchain
        for s in m_samples[np.random.randint(len(m_samples), size=100)]:
            gp.set_parameter_vector(s)
            mu = gp.predict(y, xm, return_cov=False)
            plt.plot(xm, mu, color=color, alpha=0.3)

        # Plot the data
        plt.errorbar(x, y, yerr=yerr, fmt=".k", capsize=0)
        plt.plot(xm, ym+1.0, "-", color="darkgreen", lw=2)
        plt.plot(xm, mu, color=color)
        #plt.fill_between(xm, mu+std, mu-std, color=color, alpha=0.3, edgecolor="none")
        plt.ylabel(r"CCF")
        plt.xlabel(r"Velocity [km/s]")
        plt.title("maximum likelihood prediction")
        plt.show()

        cols = mean_model.get_parameter_names()
        inds = np.array([names.index("mean:"+k) for k in cols])

        corner.corner(sampler.flatchain[:, inds], truths=bestfit_params[inds], labels=[r"$\alpha$", r"$v_0$", r"$\log\sigma^2$"])
        plt.show()

    return bestfit_params, bestfit_params_err


def best_fit_parameters(samples, verbose=False) :

    func = lambda v: (v[1], v[2]-v[1], v[1]-v[0])

    percents = np.percentile(samples, [16, 50, 84], axis=0)

    seq = list(zip(*percents))

    values = list(map(func, seq))

    mean_params, max_params, min_params = [], [], []

    for i in range(len(values)) :
        mean_params.append(values[i][0])
        max_params.append(values[i][1])
        min_params.append(values[i][2])

    min_params, max_params = np.array(min_params), np.array(max_params)
    mean_params = np.array(mean_params)
    err_params = (min_params + max_params)/2

    return mean_params, err_params


# function to interpolate spectrum
def interp_spectrum(wl_out, wl_in, flux_in, fluxerr_in, good_windows, verbose=False, plot=False) :

    flux_out = np.full_like(wl_out, np.nan)
    fluxerr_out = np.full_like(wl_out, np.nan)

    for w in good_windows :

        mask = wl_in >= w[0]
        mask &= wl_in <= w[1]
        mask &= np.isfinite(flux_in)
        mask &= np.isfinite(fluxerr_in)
        # Set up the GP model
        kernel = terms.RealTerm(log_a=np.log(np.var(flux_in[mask])), log_c=-np.log(10.0))
        gp = celerite.GP(kernel, mean=np.nanmean(flux_in[mask]), fit_mean=True)
        gp.compute(wl_in[mask], fluxerr_in[mask])
        # Fit for the maximum likelihood parameters
        initial_params = gp.get_parameter_vector()
        bounds = gp.get_parameter_bounds()
        soln = minimize(neg_log_like, initial_params, jac=grad_neg_log_like, method="L-BFGS-B", bounds=bounds, args=(flux_in[mask], gp))
        gp.set_parameter_vector(soln.x)

        wl1, wl2 = w[0], w[1]

        if wl1 < wl_in[mask][0] :
            wl1 = wl_in[mask][0]
        if wl2 > wl_in[mask][-1] :
            wl2 = wl_in[mask][-1]

        out_mask = wl_out > wl1
        out_mask &= wl_out < wl2

        flux_out[out_mask], var = gp.predict(flux_in[mask], wl_out[out_mask], return_var=True)
        fluxerr_out[out_mask] = np.sqrt(var)

    if plot :
        # Plot the data
        color = "#ff7f0e"
        plt.errorbar(wl_in, flux_in, yerr=fluxerr_in, fmt=".k", capsize=0)
        plt.plot(wl_out, flux_out, color=color)
        plt.fill_between(wl_out, flux_out+fluxerr_out, flux_out-fluxerr_out, color=color, alpha=0.3, edgecolor="none")
        plt.ylabel(r"CCF")
        plt.xlabel(r"Velocity [km/s]")
        plt.title("maximum likelihood prediction")
        plt.show()

    return flux_out, fluxerr_out
