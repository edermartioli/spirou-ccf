# -*- coding: iso-8859-1 -*-
"""
    Created on October 7 2020
    
    Description: This routine calculates the CCF data for a given data set.
    
    @author: Eder Martioli <martioli@iap.fr>
    
    Institut d'Astrophysique de Paris, France.
    
    Simple usage example:
    
    python ~/spirou-tools/spirou-ccf/spirou_ccf_pipeline.py --input=*t.fits -vo
    
    python ~/spirou-tools/spirou-ccf/spirou_ccf_pipeline.py --input=*t.fits --ccf_mask=~/spirou-tools/spirou-ccf/ccf_masks/montreal_masks/Gl905_neg.mas -vd
    
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
import spectrumlib
import ccf_lib
import ccf2rv
from copy import deepcopy

import matplotlib.pyplot as plt


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
            print("checking data file:",inputdata[i])
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
                    loc_snr = 0
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
        arg_max_snr[objectmode[0]] = np.argsort(-1*snr_per_object)
        
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


def select_apero_ccf_mask(obj_name, mask_repository) :
    
    apero_mask = mask_repository + 'montreal_masks/' + obj_name + '_neg_depth.mas'

    return apero_mask


def run_cal_ccf(efile, fp_mask, save_output=True, plot=False, verbose=False) :
    
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
    calib_ccf = ccf_lib.run_ccf_new(ccf_fp_params, spectrum_fp, rv_drifts, science_channel=False, output=save_output, plot=plot, merge_headers=True)

    return calib_ccf


def run_sci_ccf(tfile, sci_mask, source_rv=0., normalize_ccfs=True, save_output=True, plot=False, verbose=False) :

    # try to get rv drifts
    rv_drifts = get_rv_drifts(tfile, verbose)

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
    sci_ccf = ccf_lib.run_ccf_new(ccf_params, spectrum, rv_drifts,targetrv=source_rv, valid_orders=order_subset_for_mean_ccf, normalize_ccfs=normalize_ccfs, output=save_output, plot=plot, interactive_plot=False, merge_headers=True)

    return sci_ccf


def get_rv_drifts(tfits, verbose=False) :
    loc = {}
    
    fpfits = tfits.replace("t.fits","o_pp_e2dsff_C_ccf_smart_fp_mask_C.fits")
    
    if fpfits != "" and os.path.exists(fpfits) :
        if verbose :
            print("Getting RV_DRIFT from file:{}".format(fpfits))
        
        hdr = fits.getheader(fpfits,1)
        
        loc["WFPDRIFT"] = "None"
        loc["RV_WAVFP"] = hdr["RV_WAVFP"]
        loc["RV_SIMFP"] = hdr["RV_SIMFP"]
        loc["RV_DRIFT"] = hdr["RV_DRIFT"]
    
    else :
        efits = tfits.replace("t.fits","e.fits")
        
        if os.path.exists(efits) :
            print("Measuring RV_DRIFT in spectrum: {}".format(efits))
            try :
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
            except:
                print("WARNING: could not calculate RV drift on file {0}, setting RV drift = 0.".format(efits))
                loc["WFPDRIFT"] ='None'
                loc["RV_WAVFP"] = 'None'
                loc["RV_SIMFP"] = 'None'
                loc["RV_DRIFT"] = 0.
        else :
            print("WARNING: could not find {} nor {}, setting RV drift = 0.".format(fpfits, efits))
            loc["WFPDRIFT"] ='None'
            loc["RV_WAVFP"] = 'None'
            loc["RV_SIMFP"] = 'None'
            loc["RV_DRIFT"] = 0.

    return loc


def process_ccfs(file_list, mask_file, source_rv=0., correct_rv_drift=False, rv_file="", overwrite=False, verbose=False) :
    
    bjd, rv, rverr = [], [], []
    sci_ccf_file_list, valid_files = [], []
    snr = []

    for i in range(len(file_list)) :
        try :
            outpath = ccf_lib.construct_out_ccf_filename(file_list[i], mask_file)

            if not os.path.exists(outpath) or overwrite :
                if verbose :
                    print("Running CCF on file {0}/{1}:{2}".format(i,len(file_list)-1,os.path.basename(file_list[i])))

                sci_ccf = run_sci_ccf(file_list[i], mask_file, source_rv=source_rv)

                if verbose:
                    print("Spectrum: {0} DATE={1} SNR={2:.0f} Sci_RV={3:.5f} km/s RV_DRIFT={4:.5f} km/s".format(os.path.basename(file_list[i]), sci_ccf["header"]["DATE"],sci_ccf["header"]["SPEMSNR"], sci_ccf["header"]['RV_OBJ'], sci_ccf["header"]["RV_DRIFT"]))

                bjd.append(sci_ccf["header"]['BJD'] + (sci_ccf["header"]['MJDEND'] - sci_ccf["header"]['MJDATE'])/2.)

                if correct_rv_drift :
                    rv.append(sci_ccf["header"]['RV_OBJ'] - sci_ccf["header"]["RV_DRIFT"])
                else :
                    rv.append(sci_ccf["header"]['RV_OBJ'])

                rverr.append(sci_ccf["header"]['CCFMRVNS'])
                snr.append(sci_ccf["header"]["SPEMSNR"])
            
                sci_ccf_file_list.append(os.path.abspath(sci_ccf["file_path"]))
                valid_files.append(file_list[i])
                    
            elif os.path.exists(outpath) :
                
                hdr = fits.getheader(outpath, 1)
                
                if verbose:
                    print("CCF from existing file: {0} DATE={1} SNR={2:.0f} Sci_RV={3:.5f} km/s RV_DRIFT={4:.5f} km/s".format(os.path.basename(outpath), hdr["DATE"],hdr["SPEMSNR"], hdr['RV_OBJ'], hdr["RV_DRIFT"]))

                bjd.append(hdr['BJD'] + (hdr['MJDEND'] - hdr['MJDATE'])/2.)

                if correct_rv_drift :
                    rv.append(hdr['RV_OBJ'] - hdr["RV_DRIFT"])
                else :
                    rv.append(hdr['RV_OBJ'])

                rverr.append(hdr['CCFMRVNS'])
                snr.append(hdr["SPEMSNR"])
                    
                # consider only valid files those that exist
                sci_ccf_file_list.append(outpath)
                valid_files.append(file_list[i])
                    
        except Exception as e:
            print("WARNING: could not run CCF on file {}, skipping ... ".format(file_list[i]))
            print(e)

    bjd = np.array(bjd)
    rv, rverr = np.array(rv), np.array(rverr)
    snr = np.array(snr)

    if rv_file != "" :
        if verbose :
            print("Saving output RVs in the file: {0}".format(rv_file))
        spiroulib.save_rv_time_series(rv_file, bjd, rv, rverr)

    loc = {}
    loc["sci_ccf_files"] = sci_ccf_file_list
    loc["valid_files"] = valid_files
    loc["bjd"] = bjd
    loc["rv"] = rv
    loc["rverr"] = rverr
    loc["snr"] = snr
    return loc


def calculate_optimum_ccf_mask(template_spectrum, ref_sci_ccf, obj_temp, vsini=2., mask_file="", output="", verbose=False, plot=False) :

    #############
    ref_fwhm = ref_sci_ccf['MEAN_FWHM']
    # Need to create a tool to measure vsini or use empirical CCF to detect lines
    ccf_width, ccf_step = 2.5 * ref_fwhm, ref_sci_ccf['CCF_STEP']
    spirou_resolution = 70000
    if verbose :
        print("ref_fwhm=",ref_fwhm, "ccf_width=",ccf_width, "ccf_step=",ccf_step)

    vald_database = spirou_ccf_dir + 'VALD.950_2400nm/atomic_lines.tsv'

    if verbose :
        print("Detecting lines in template ...")
    empirical_lines = spectrumlib.detect_lines_in_template(template_spectrum, ccf_width, ccf_step, spirou_resolution, vsini, plot=plot)

    # save and read empirical lines  

    if mask_file != "":
        # load an input ccf  mask for comparison
        linemask = {}
        linemask = spectrumlib.get_ccf_mask(linemask, mask_file)
    else :
        linemask = None

    #mask_width=ref_sci_ccf["MASK_WIDTH"]
    mask_width=1.0
    if verbose :
        print("Creating mask of empirical lines ...")
    catalog = spectrumlib.make_mask_of_empirical_lines(empirical_lines, mask_width=mask_width, sig_clip=5.0, min_fwhm=0.25*ref_fwhm, max_fwhm=3*ref_fwhm, use_measured_wlc=False, linemask=linemask, output=output, include_orders_in_mask=False, verbose=True, plot=True)
    
    """
    if verbose :
        print("Identifying detected lines with VALD database...")
    catalog = spectrumlib.identify_detected_lines(empirical_lines, vald_database, ccf_step, obj_temp, linemask=linemask, plot=False, verbose=False)
    
    #rot_ccf = spectrumlib.get_rot_ccf(v_min=-ccf_width, v_max=ccf_width, v_step=ccf_step, resolution=spirou_resolution, wlc=1600., vsini=vsini, teff=obj_temp, plot=False)
    #ref_fwhm = rot_ccf['fwhm']
    #print("ref_fwhm=",ref_fwhm)

    if options.verbose :
        print("Generating optimum CCF mask and saving to file: {}".format(optimum_mask))

    ccf_mask = spectrumlib.generate_ccf_optimal_mask(catalog, mask_width=mask_width, sig_clip=2.5, outputmask=output, include_orders_in_mask=False, min_fwhm=0.25*ref_fwhm, max_fwhm=3*ref_fwhm, use_measured_wlc=True, plot=plot)
    
    return ccf_mask
    """
    return catalog


def run_ccf_analysis(ccf_files, mask_file, obj="", drs_version="", snr_min=20., dvmax_per_order=3.0, sanit=False, correct_rv_drift=False, save_ccf_fitsfile=False, plot=False, verbose=False) :
    ##########################
    # Run CCF analysis to get object rvs using an optimized algorithm
    mask_basename = os.path.basename(mask_file)
    
    collection_key = "{}__{}__{}__{}".format(obj, mask_basename, sanit, drs_version)
    
    if verbose :
        print("Running CCF analysis on CCF data for collection: {}".format(collection_key))
    
    # exclude orders with strong telluric absorption
    #exclude_orders = [-1]  # to include all orders
    #exclude_orders = [11,12,22,23,24,25,26,27,37,38,39,40,47,48]
    exclude_orders = [0,1,10,11,12,13,14,21,22,23,24,25,26,27,37,38,39,40,47,48]

    # set bandpass
    bandpass = "YJHK"

    # form a unique batch name with mask, object and method
    batch_name = '{0}/{1}__{2}'.format(outdir, collection_key, bandpass)

    ccf = ccf2rv.get_object_rv(ccf_files, collection_key=collection_key, method="all", exclude_orders=exclude_orders, snr_min=snr_min, bandpass=bandpass, dvmax_per_order=dvmax_per_order, save_rdb_timeseries=True, correct_rv_drift=correct_rv_drift, save_csv_table_of_results=True, save_ccf_cube=False, save_weight_table=False, doplot=plot, showplots=plot, save_ccf_fitsfile=save_ccf_fitsfile, saveplots=plot, detailed_output=True, verbose=verbose)

    return ccf
    ##########################


parser = OptionParser()
parser.add_option("-i", "--input", dest="input", help="Spectral *t.fits data pattern",type='string',default="*t.fits")
parser.add_option("-m", "--ccf_mask", dest="ccf_mask", help="Input CCF mask",type='string',default="")
parser.add_option("-e", "--ref_spectrum", dest="ref_spectrum", help="Input reference spectrum",type='string',default="")
parser.add_option("-r", "--source_rv", dest="source_rv", help="Input source RV [km/s]",type='float',default=0.0)
parser.add_option("-s", action="store_true", dest="stack_polar", help="Stack polar sequence", default=False)
parser.add_option("-t", action="store_true", dest="run_template", help="Run template", default=False)
parser.add_option("-d", action="store_true", dest="correct_drift", help="correct RV drift", default=False)
parser.add_option("-o", action="store_true", dest="overwrite", help="overwrite output files", default=False)
parser.add_option("-p", action="store_true", dest="plot", help="plot", default=False)
parser.add_option("-v", action="store_true", dest="verbose", help="verbose", default=False)

try:
    options,args = parser.parse_args(sys.argv[1:])
except:
    print("Error: check usage with  -h spirou_ccf_pipeline.py")
    sys.exit(1)

if options.verbose:
    print('Spectral t.fits data pattern: ', options.input)
    if options.ccf_mask != "":
        print('Input CCF mask: ', options.ccf_mask)
    if options.ref_spectrum != "":
        print('Input reference spectrum: ', options.ref_spectrum)
    if options.source_rv :
        print('Input source RV [km/s]: ', options.source_rv)

correct_rv_drift = options.correct_drift

# make list of tfits data files
if options.verbose:
    print("Creating list of t.fits spectrum files...")
inputdata = sorted(glob.glob(options.input))

if options.stack_polar :
    polar_sets = spiroulib.generate_polar_sets(inputdata, verbose=options.verbose)

    inputdata = spiroulib.stack_polar_sequence(polar_sets, correct_drift=options.correct_drift, overwrite=options.overwrite)

# First bunch up all input spectra of the same object and check type of data
collections, objtemps, arg_max_snr = generate_collection(inputdata, verbose=True)

for obj in collections['object'] :
    
    file_list = collections[obj]
    obj_temp = objtemps[obj]
    
    if options.ccf_mask != "" :
        # Force to use an input mask
        mask_file = os.path.normpath(options.ccf_mask)
    else :
        # Select mask that best matches object temperature
        mask_file = select_best_ccf_mask(obj_temp, mask_repository)
    
    if options.verbose :
        print("*************************************************")
        print("OBJECT: {0} TEFF: {1}K  REF_CCF_MASK: {2}".format(obj, obj_temp,os.path.basename(mask_file)))
        print("*************************************************")

    ref_not_processed = True

    if options.ref_spectrum != "" :
        refexp = options.ref_spectrum
        ref_sci_ccf = run_sci_ccf(refexp, mask_file, source_rv=options.source_rv, save_output=False, plot=options.plot)

        try :
            if options.verbose :
                print("Trying to run CCF on reference exposure:{}".format(os.path.basename(refexp)))
            # Run ccf on the reference spectrum, i.e., the one with maximum SNR:
            ref_sci_ccf = run_sci_ccf(refexp, mask_file, source_rv=options.source_rv, save_output=False, plot=options.plot)
            
            ref_not_processed = False
        except :
            print("WARNING: could not select file {} as reference, skipping ...".format(refexp))

                
    if ref_not_processed :
        
        file_list_sorted_by_snr = file_list[arg_max_snr[obj]]

        # loop over the list of input files sorted by SNR, and pick the first possible ref file
        for i in range(len(file_list_sorted_by_snr)) :
            #try :
            if 0 == 0 :
                #set reference exposure as the one with maximum SNR
                refexp = file_list_sorted_by_snr[i]

                if options.verbose :
                    print("Trying to run CCF on reference exposure:{}".format(os.path.basename(refexp)))

                # Run ccf on the reference spectrum, i.e., the one with maximum SNR:
                ref_sci_ccf = run_sci_ccf(refexp, mask_file, source_rv=options.source_rv, save_output=False, plot=options.plot)

                ref_not_processed = False
                # get out of the loop when succeed to calculate CCF
                break
            #except :
            #    print("WARNING: could not select file {} as reference, skipping ...".format(refexp))
            #    continue

    if ref_not_processed :
        print("ERROR: could not select any file as reference, exiting ...")
        exit()
    
    # Set path for output files
    abs_path_of_ref_spectrum = os.path.abspath(refexp)
    outdir = os.path.dirname(abs_path_of_ref_spectrum)

    if options.source_rv :
        # Set systemic velocity to the input source RV
        rv_sys = options.source_rv
    else :
        # Set systemic velocity to the RV measured on the reference exposure
        rv_sys = ref_sci_ccf["header"]['RV_OBJ']

    if options.verbose :
        print("Reference spectrum {0} observed on {1} SNR={2} showing RV_sys={3:.5f} km/s".format(os.path.basename(refexp), ref_sci_ccf["header"]["DATE"], ref_sci_ccf["header"]["SPEMSNR"], rv_sys))

    if options.ccf_mask == "" :
        # Select best mask based on measured RV sys
        mask_file = select_best_ccf_mask(obj_temp, mask_repository, rv_sys=rv_sys)
        if options.verbose :
            print("Selected optimal CCF MASK: {0}".format(mask_file))

    ##########################
    # Run CCF:
    # Set path and filename for output template
    mask_basename = os.path.basename(mask_file)
    rv_file = outdir + "/{0}_{1}_rv.rdb".format(obj,mask_basename)
    # process CCFs for all spectra using best mask from mask collection
    pccfs = process_ccfs(file_list, mask_file, source_rv=rv_sys, correct_rv_drift=correct_rv_drift, rv_file=rv_file, overwrite=options.overwrite, verbose=options.verbose)
    ##########################

    ########################
    dvmax_per_order = 3*np.nanstd(pccfs["rv"])
    snr_min = np.nanmedian(pccfs["snr"]) - 5*np.nanstd(pccfs["snr"])
    if snr_min < 20. : snr_min = 20

    if options.verbose :
        print("Running CCF analysis with dvmax_per_order={0:.3f} km/s  snr_min={1:.0f}",dvmax_per_order, snr_min)
    # Run CCF analysis on CCF data calculated previously to obtain optimal RVs
    ccf_results = run_ccf_analysis(pccfs["sci_ccf_files"], mask_file, obj=obj, drs_version=fits.getheader(refexp)['VERSION'], snr_min=snr_min, dvmax_per_order=dvmax_per_order, correct_rv_drift=correct_rv_drift, plot=options.plot, verbose=options.verbose)
    #######################

    if options.run_template :
        ##########################
        # Create template spectrum
        template_output = outdir + "/{}_template.fits".format(obj)

        if not os.path.exists(template_output) or options.overwrite :
            if options.verbose :
                print("Building template spectrum out of {0} spectra ...".format(len(pccfs["valid_files"])))
            try :
                # Build template spectrum for object:
                template_spectrum = spiroulib.template_using_fit(pccfs["valid_files"], rv_file, median=True, normalize_by_continuum=True, verbose=False, plot=options.plot)
            
                # Set path and filename for output template
                if options.verbose :
                    print("Saving template spectrum in the file: {0} ".format(template_output))
                spiroulib.write_spectrum_orders_to_fits(template_spectrum, template_output, header=fits.getheader(refexp))
            except :
                if options.verbose :
                    print("WARNING: could not create template, adopting template as ref spectrum:  {0} ".format(refexp))
                template_spectrum = spiroulib.load_spirou_AB_efits_spectrum(refexp, nan_pos_filter=False, preprocess=True, source_rv=rv_sys, normalization_in_preprocess=2, normalize_blaze=True)
        else :
            print("Template already exists, loading spectrum from file: {0}".format(template_output))
            # Load template spectrum
            template_spectrum = spiroulib.read_orders_spectrum_from_fits(template_output)

        ##########################
        # Run routines to calculate an optimum mask for object using template:
        # Set filename for output optimum mask
        #optimum_mask = outdir + "/{0}.mas".format(obj)
        #if not os.path.exists(optimum_mask) or options.overwrite :
        #    ccf_mask = calculate_optimum_ccf_mask(template_spectrum, ref_sci_ccf, obj_temp, mask_file=mask_file, output=optimum_mask, verbose=options.verbose)
        ##########################








