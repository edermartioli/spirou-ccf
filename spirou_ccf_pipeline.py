# -*- coding: iso-8859-1 -*-
"""
    Created on October 7 2020
    
    Description: This routine calculates the CCF data for a given data set.
    
    @author: Eder Martioli <martioli@iap.fr>
    
    Institut d'Astrophysique de Paris, France.
    
    Simple usage example:
    
    python ~/spirou-ccf/spirou_ccf_pipeline.py --input=*t.fits
    
    """

__version__ = "1.0"

__copyright__ = """
    Copyright (c) ...  All rights reserved.
    """

from optparse import OptionParser
import os,sys
import astropy.io.fits as fits
import glob
from collections import Counter
import numpy as np

import spiroulib
import ccf_lib
import ccf2rv

spirou_ccf_dir = os.path.dirname(__file__) + '/'

mask_repository = spirou_ccf_dir + 'ccf_masks/'

fp_mask = mask_repository + 'fp.mas'


def generate_collection(inputdata, verbose=False) :
    """
        Description: function to detect and organize collection of files
                     based on: object name
        inputdata: list, list of file paths
        return: dict, where:
            dict["modes"] : list of keys to identify collections
            dict["key"] : list of files for a given collection
        """
    object_names = []
    temperatures = []
    snr = []
    valid_files = []

    for i in range(len(inputdata)) :
        try :
            hdr = fits.getheader(inputdata[i], 0)
            combhdr = fits.getheader(inputdata[i], 0) + fits.getheader(inputdata[i], 1)

            filename = hdr['FILENAME']
            if 'DRSOBJN' in hdr.keys() :
                object_name = hdr['DRSOBJN'].replace(" ","").upper()
            elif 'OBJECT' in hdr.keys() :
                object_name = hdr['OBJECT'].replace(" ","").upper()
            else :
                # if object names as not available, try to bunch up by RA and DEC
                object_name = "RA{0}_DEC{1}".format(hdr["RA"],hdr["DEC"])
            
            # *e.fits files
            if inputdata[i].endswith("t.fits" or "e.fits") :
                
                if 0 < hdr["OBJTEMP"] < 1e6 :
                    temperatures.append(hdr["OBJTEMP"])
                else :
                    temperatures.append(np.nan)
            
                if "EXTSN035" in combhdr.keys() :
                    loc_snr = float(combhdr["EXTSN035"])
                elif "SNR35" in combhdr.keys():
                    loc_snr = float(combhdr["SNR35"])
                else :
                    loc_snr = np.nan
                snr.append(loc_snr)

                object_names.append(object_name)
                valid_files.append(os.path.abspath(inputdata[i]))
            else :
                if verbose :
                    print("WARNING: unsupported suffix for file",inputdata[i], ", skippping ... ")
                continue
        except :
            if verbose :
                print("WARNING: could not read file",inputdata[i], ", skippping ... ")
            continue

    temperatures = np.array(temperatures)
    snr = np.array(snr)
    object_names = np.array(object_names)
    valid_files = np.array(valid_files)
    
    modes = Counter(object_names)
    most_common_objects = modes.most_common()   # Returns all unique items and their counts
    
    collection = {}
    objtemp = {}
    arg_max_snr = {}
    collection["object"] = []
    
    for objectmode in most_common_objects :
        # create mask to select files within the same mode
        object_mask = object_names == objectmode[0]
        collection["object"].append(objectmode[0])
        collection[objectmode[0]] = valid_files[object_mask]
        
        # figure out most common object temperature that is not NaN
        temps_per_object = temperatures[object_mask]
        nanmask = ~np.isnan(temps_per_object)
        if len(temps_per_object[nanmask]) :
            b = Counter(temps_per_object[nanmask])
            objtemp[objectmode[0]] = b.most_common()[0][0]
        else :
            objtemp[objectmode[0]] = np.nan

        snr_per_object = snr[object_mask]
        snr_nanmask = ~np.isnan(snr_per_object)
        if len(snr_per_object[snr_nanmask]) :
            arg_max_snr[objectmode[0]] = np.nanargmax(snr_per_object)
        else :
            arg_max_snr[objectmode[0]] = 0
        
        if verbose:
            print("Collection name:{0} OBJTEMP={1} NFILES={2}".format(objectmode[0],objtemp[objectmode[0]], len(collection[objectmode[0]])))

    return collection, objtemp, arg_max_snr



def select_best_ccf_mask(obj_temp, mask_repository, rv_sys=-99999.) :
    
    default_mask = mask_repository + 'masque_sept18_andres_trans50.mas'
    
    if rv_sys == -99999. :
        #pattern = mask_repository + '*_all.mas'
        pattern = mask_repository + '*_filt_allRV.mas'
    else :
        pattern = mask_repository + '*weighted_RV*.mas'
    
    list_of_masks = sorted(glob.glob(pattern))

    teff = {'K2':5000, 'M2':3500, 'M4': 3100, 'M5':2900}

    best_mask = default_mask
    temp_diff = 1e20
    rv_diff = 1e20
    
    for mask in list_of_masks :
        
        # get rid of full path before file name
        mask_basename = os.path.basename(mask)
        
        # get spectral type of mask
        mask_spectral_type = mask_basename[:2]

        
        if mask_spectral_type in teff.keys() :
            mask_temp = teff[mask_spectral_type]
        else :
            print("Mask spectral type {0} not recognized, skipping file {1}".format(mask_spectral_type,mask))
            continue
        loc_temp_diff = np.abs(obj_temp - mask_temp)


        if rv_sys != -99999. :
            try :
                # get sys rv of mask
                mask_rv = float(mask_basename.split('.')[0].split("_")[-1])
                if mask_rv < -120. or mask_rv > +120.:
                    print("Mask RV {0} out of range, skipping file {1}".format(mask_rv,mask))
                    continue
            except :
                print("WARNING: Mask RV is not a float, file name formating not recognized, skipping file {0}".format(mask))
                continue
            loc_rv_diff = np.abs(rv_sys - mask_rv)

            if loc_temp_diff < temp_diff or loc_rv_diff < rv_diff:
                temp_diff = loc_temp_diff
                rv_diff = loc_rv_diff
                best_mask = mask

        else :
            if loc_temp_diff < temp_diff :
                temp_diff = loc_temp_diff
                best_mask = mask

    if rv_sys == -99999. :
        if temp_diff == 1e20 :
            print("WARNING: Could not identify any suitable mask, returning default mask: {}".format(best_mask))
    else :
        if temp_diff == 1e20 or rv_diff == 1e20:
            print("WARNING: Could not identify any suitable mask, returning default mask: {}".format(best_mask))

    return best_mask


def run_cal_ccf(efile, fp_mask, plot=False, verbose=False) :
    
    rv_drifts = {}
    rv_drifts["WFPDRIFT"] ='None'
    rv_drifts["RV_WAVFP"] = 'None'
    rv_drifts["RV_SIMFP"] = 'None'
    rv_drifts["RV_DRIFT"] = 0.
    ########################################
    # Then run CCF on the calibration fiber:
    ########################################
    if verbose :
        print("Running CCF on the calibration fiber")
    # Load SPIRou reduced e FITS file -- to obtain Fiber-C FP spectrum
    spectrum_fp = spiroulib.load_spirou_AB_efits_spectrum(efile, nan_pos_filter=False, normalize_blaze=False)
    # load calibration CCF parameters
    ccf_fp_params = ccf_lib.set_ccf_params(fp_mask, science_channel=False)
    # run main routine to process ccf on calibration fiber
    calib_ccf = ccf_lib.run_ccf_new(ccf_fp_params, spectrum_fp, rv_drifts, science_channel=False, plot=plot, merge_headers=True)

    return calib_ccf


def run_sci_ccf(tfile, sci_mask, plot=False, verbose=False) :

    # try to get rv drifts
    rv_drifts = get_rv_drifts(tfile)

    ####################################
    # First run CCF on the science fiber:
    ####################################
    if verbose :
        print("Running CCF on the science fiber")
    order_subset_for_mean_ccf = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 13, 14, 15, 16, 17, 18, 19, 20, 21, 28, 29, 30, 31, 32, 33, 34, 35, 43, 44, 45, 46]
    # Load SPIRou reduced *t.fits file
    spectrum = spiroulib.load_spirou_AB_efits_spectrum(tfile, nan_pos_filter=False, normalize_blaze=False)
    # load science CCF parameters
    ccf_params = ccf_lib.set_ccf_params(sci_mask)
    # run main routine to process ccf on science fiber
    sci_ccf = ccf_lib.run_ccf_new(ccf_params, spectrum, rv_drifts, valid_orders=order_subset_for_mean_ccf, plot=plot, interactive_plot=False, merge_headers=True)

    return sci_ccf


def get_rv_drifts(tfits) :
    loc = {}
    
    abspath = os.path.abspath(tfile)
    tfilebasename = os.path.basename(tfile)
    
    fpfits = abspath +'/'+tfilebasename.replace("t.fits","_pp_e2dsff_C_ccf_smart_fp_mask_C.fits")
    efits = abspath +'/'+tfilebasename.replace("t.fits","e.fits")
    
    print(fpfits, efits)
    
    if os.path.exists(fpfits) :
        print("Getting RV_DRIFT from file:{}".format(fpfits))
        
        hdr = fits.getheader(fpfits)
        
        loc["WFPDRIFT"] = hdr["WFPDRIFT"]
        loc["RV_WAVFP"] = hdr["RV_WAVFP"]
        loc["RV_SIMFP"] = hdr["RV_SIMFP"]
        loc["RV_DRIFT"] = hdr["RV_DRIFT"]
    
    else :

        if os.path.exists(efits) :
            print("Measuring RV_DRIFT in spectrum: {}".format(efits))
            calib_ccf = run_cal_ccf(efits, fp_mask, plot=False, verbose=False)
            hdr = fits.getheader(tfits)
            if "WFPDRIFT" in hdr.keys() :
                loc["WFPDRIFT"] = hdr["WFPDRIFT"]
            else :
                loc["WFPDRIFT"] = 'None'

            if "WFPDRIFT" in hdr.keys() :
                loc["RV_WAVFP"] = hdr["RV_WAVFP"]
            else :
                loc["RV_WAVFP"] = 'None'

            loc["RV_SIMFP"] = calib_ccf["header"]['RV_OBJ']
            loc["RV_DRIFT"] = calib_ccf["header"]['RV_OBJ']
            
        else :
            print("WARNING: could not find {} nor {}, setting RV drift = 0.".format(fpfits, efits))
            loc["WFPDRIFT"] ='None'
            loc["RV_WAVFP"] = 'None'
            loc["RV_SIMFP"] = 'None'
            loc["RV_DRIFT"] = 0.

    return loc


parser = OptionParser()
parser.add_option("-i", "--input", dest="input", help="Spectral *t.fits data pattern",type='string',default="*t.fits")
parser.add_option("-s", action="store_true", dest="save_template", help="save template spectrum to  OBJECT_template.fits", default=False)
parser.add_option("-p", action="store_true", dest="plot", help="plot", default=False)
parser.add_option("-v", action="store_true", dest="verbose", help="verbose", default=False)

try:
    options,args = parser.parse_args(sys.argv[1:])
except:
    print("Error: check usage with  -h spirou_ccf_pipeline.py")
    sys.exit(1)

if options.verbose:
    print('Spectral t.fits data pattern: ', options.input)

correct_rv_drift = True

# make list of tfits data files
if options.verbose:
    print("Creating list of t.fits spectrum files...")
inputdata = sorted(glob.glob(options.input))

# First bunch up all input spectra of the same object and check type of data
collections, objtemps, arg_max_snr = generate_collection(inputdata, verbose=True)

for object in collections['object'] :
    
    file_list = collections[object]
    obj_temp = objtemps[object]
    # set reference exposure as the one with maximum SNR
    refexp = arg_max_snr[object]
    
    # Select mask that best matches object temperature
    mask_file = select_best_ccf_mask(obj_temp, mask_repository)

    # Set path for output files
    abs_path_of_ref_spectrum = os.path.abspath(file_list[refexp])
    outdir = os.path.dirname(abs_path_of_ref_spectrum)
    
    if options.verbose :
        print("*************************************************")
        print("OBJECT: {0} TEFF: {1}K  REF_CCF_MASK: {2}".format(object, obj_temp,os.path.basename(mask_file)))
        print("*************************************************")

    if options.verbose :
        print("Running CCF on reference exposure:{}".format(os.path.basename(file_list[refexp])))
    # Run ccf on the reference spectrum, i.e., the one with maximum SNR:
    ref_sci_ccf = run_sci_ccf(file_list[refexp], mask_file, plot=options.plot)

    # Set systemic velocity to the RV measured on the reference exposure
    rv_sys = ref_sci_ccf["header"]['RV_OBJ']

    if options.verbose :
        print("Reference spectrum {0} observed on {1} showing RV_sys={2:.5f} km/s".format(file_list[refexp], ref_sci_ccf["header"]["DATE"], rv_sys))

    # Select best mask based on measured RV sys
    mask_file = select_best_ccf_mask(obj_temp, mask_repository, rv_sys=rv_sys)
    if options.verbose :
        print("Selected optimal CCF MASK: {0}".format(mask_file))

    bjd, rv, rverr = [], [], []
    sci_ccf_file_list = []
    snr = []

    for i in range(len(file_list)) :
        if options.verbose :
            print("Running CCF on file {0}/{1}:{2}".format(i,len(file_list)-1,os.path.basename(file_list[i])))

        sci_ccf = run_sci_ccf(file_list[i], mask_file)

        if options.verbose:
            print("Spectrum: {0} DATE={1} SNR={2:.0f} Sci_RV={3:.5f}km/s RV_DRIFT={4}km/s".format(os.path.basename(file_list[i]), sci_ccf["header"]["DATE"],sci_ccf["header"]["SPEMSNR"], sci_ccf["header"]['RV_OBJ'], sci_ccf["header"]["RV_DRIFT"]))

        sci_ccf_file_list.append(os.path.abspath(sci_ccf["file_path"]))
        
        bjd.append(sci_ccf["header"]['BJD'] + (sci_ccf["header"]['MJDEND'] - sci_ccf["header"]['MJDATE'])/2.)

        if correct_rv_drift :
            rv.append(sci_ccf["header"]['RV_OBJ'] - sci_ccf["header"]["RV_DRIFT"])
        else :
            rv.append(sci_ccf["header"]['RV_OBJ'])

        rverr.append(sci_ccf["header"]['CCFMRVNS'])
        snr.append(sci_ccf["header"]["SPEMSNR"])

    bjd = np.array(bjd)
    rv, rverr = np.array(rv), np.array(rverr)
    snr = np.array(snr)
    
    # Set path and filename for output template
    rv_file = outdir + "/{0}_rv.rdb".format(object)
    if options.verbose :
        print("Saving output RVs in the files: {0}".format(rv_file))
    spiroulib.save_rv_time_series(rv_file, bjd, rv, rverr)

    if options.verbose :
        print("Building template spectrum out of {0} spectra ...".format(len(file_list)))
    try :
        # Build template spectrum for object:
        template_spectrum = spiroulib.template_using_fit(file_list, rv_file, median=True, normalize_by_continuum=True, verbose=False, plot=options.plot)

        if options.save_template :
            # Set path and filename for output template
            template_output = outdir + "/{}_template.fits".format(object)
            if options.verbose :
                print("Saving template spectrum in the file: {0} ".format(template_output))
            spiroulib.write_spectrum_to_fits(template_spectrum, template_output, header=fits.getheader(file_list[refexp]), wavekey='wl_template', fluxkey='flux_template', fluxerrkey='fluxerr_template')
    except Exception as e:
        print("WARNING: could not create template, skipping ... ")
        print(e)

    # Run CCF analysis to get object rvs using an optimized algorithm
    mask_basename = os.path.basename(mask_file)
    sanit, drs_version = False, fits.getheader(file_list[refexp])['VERSION']
    collection_key = "{}__{}__{}__{}".format(object, mask_basename, sanit, drs_version)

    if options.verbose :
        print("Running CCF analysis on CCF data for collection: {}".format(collection_key))

    # exclude orders with strong telluric absorption
    #exclude_orders = [-1]  # to include all orders
    exclude_orders = [11,12,22,23,24,25,26,27,37,38,39,40,47,48]

    # Set minimum SNR.
    #snr_min = np.nanmedian(snr) - 3. * np.nanstd(snr)
    #if snr_min < 1 : snr_min = 1.
    snr_min=20.

    # set bandpass
    bandpass = "YJHK"

    tbl = ccf2rv.get_object_rv(sci_ccf_file_list, collection_key=collection_key, method="all", exclude_orders=exclude_orders, snr_min=snr_min, bandpass=bandpass, save_rdb_timeseries=True, correct_rv_drift=correct_rv_drift, save_csv_table_of_results=True, save_ccf_cube=False, save_weight_table=False, doplot=options.plot, showplots=options.plot, saveplots=options.plot, verbose=options.verbose)

    # To do:

    # Check FP mask, FP RV calculations

    # Use template to build optimal CCF mask for object
    # Calculate CCFs using optimal mask on a second pass

    # Run algorithm to compare all RV data and pick the one with minimum RV dispersion


