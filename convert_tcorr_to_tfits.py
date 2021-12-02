# -*- coding: iso-8859-1 -*-
"""
    Created on Sep 13 2021
    
    Description: This routine is a patch that converts tcorr/e2ds files into t.fits/e.fits by replacing the data and keywords in an existing e/t.fits by the data/keywords in the tcorr/e2ds files. This tool was created to convert data reduced in MTL which don't produce the e/t.fits files.
    
    @author: Eder Martioli <martioli@iap.fr>
    
    Institut d'Astrophysique de Paris, France.
    
    Simple usage example:
    python ~/spirou-tools/spirou-ccf/convert_tcorr_to_tfits.py --tcorr=/Volumes/EDERIAP/SLS-DATA/GL699/tcorr/2*tcorr_AB.fits --tfits=/Volumes/EDERIAP/SLS-DATA/GL699/2*t.fits -v
    
    
    python ~/spirou-tools/spirou-ccf/convert_tcorr_to_tfits.py --tcorr=/Volumes/EDERIAP/SLS-DATA/TOI-1759/MTL/2*tcorr_AB.fits --tfits=/Volumes/EDERIAP/SLS-DATA/TOI-1759/2*t.fits -v
    
    python ~/spirou-tools/spirou-ccf/convert_tcorr_to_tfits.py --tcorr=/Volumes/EDERIAP/SLS-DATA/TOI-1759/MTL/2*_e2dsff_AB.fits --tfits=/Volumes/EDERIAP/SLS-DATA/TOI-1759/2*e.fits -v
    """

__version__ = "1.0"

__copyright__ = """
    Copyright (c) ...  All rights reserved.
    """

from optparse import OptionParser
import os,sys

import glob

import matplotlib.pyplot as plt
import numpy as np
import astropy.io.fits as fits


def fits2wave(file_or_header, npix=0):
    info = """
        Provide a fits header or a fits file
        and get the corresponding wavelength
        grid from the header.
        
        Usage :
        wave = fits2wave(hdr)
        or
        wave = fits2wave('my_e2ds.fits')
        
        Output has the same size as the input
        grid. This is derived from NAXIS
        values in the header
        """
    
    
    # check that we have either a fits file or an astropy header
    if type(file_or_header) == str:
        hdr = fits.getheader(file_or_header)
    elif str(type(file_or_header)) == "<class 'astropy.io.fits.header.Header'>":
        hdr = file_or_header
    else:
        print()
        print('~~~~ wrong type of input ~~~~')
        print()
        
        print(info)
        return []

    # get the keys with the wavelength polynomials
    wave_hdr = hdr['WAVE0*']
    # concatenate into a numpy array
    wave_poly = np.array([wave_hdr[i] for i in range(len(wave_hdr))])
    
    # get the number of orders
    nord = hdr['WAVEORDN']
    
    # get the per-order wavelength solution
    wave_poly = wave_poly.reshape(nord, len(wave_poly) // nord)
    
    # get the length of each order  if not provided (normally that's 4088 pix)
    if npix == 0 :
        npix = hdr['NAXIS1']
    
    # project polynomial coefficiels
    wavesol = [np.polyval(wave_poly[i][::-1],np.arange(npix)) for i in range(nord) ]
    
    # return wave grid
    return np.array(wavesol)


parser = OptionParser()
parser.add_option("-c", "--tcorr", dest="tcorr", help="Spectral tcorr fits data pattern",type='string',default="*tcorr.fits")
parser.add_option("-t", "--tfits", dest="tfits", help="Spectral t.fits data pattern",type='string',default="*t.fits")
parser.add_option("-k", action="store_true", dest="keepwave", help="to keep wavelength in t.fits", default=False)
parser.add_option("-v", action="store_true", dest="verbose", help="verbose", default=False)
parser.add_option("-p", action="store_true", dest="plot", help="plot", default=False)

try:
    options,args = parser.parse_args(sys.argv[1:])
except:
    print("Error: check usage with  -h convert_tcorr_to_tfits.py")
    sys.exit(1)

if options.verbose:
    print('Spectral tcorr/e2ds fits data pattern: ', options.tcorr)
    print('Spectral t.fits data pattern: ', options.tfits)

# make list of efits data files
if options.verbose:
    print("Creating list of tcorr fits spectrum files...")
tcorrdata = sorted(glob.glob(options.tcorr))

# make list of tfits data files
if options.verbose:
    print("Creating list of t.fits spectrum files...")
inputtdata = sorted(glob.glob(options.tfits))

lastj = 0
nmatch = 0
for i in range(len(tcorrdata)) :

    odo = os.path.basename(tcorrdata[i])[:7]
    
    for j in range(lastj,len(inputtdata)) :
        tfits_odo = os.path.basename(inputtdata[j])[:7]
        
        if tfits_odo == odo :
        #if True :
            if inputtdata[j].endswith("t.fits") :
                output_name = inputtdata[j].replace("t.fits","t_tcorr.fits")
            elif inputtdata[j].endswith("e.fits") :
                output_name = inputtdata[j].replace("e.fits","e_e2ds.fits")
            else :
                print("ERROR: file extension must be t.fits or e.fits. Skipping ...")
                continue
            
            print("Found matching odometer: ", odo," converting data and saving to:",output_name)

            tfits_hdu = fits.open(inputtdata[j])
            tcorr_hdu = fits.open(tcorrdata[i])

            tfits_hdu["FluxAB"].data = tcorr_hdu[0].data
            if not options.keepwave :
                tfits_hdu["WaveAB"].data = fits2wave(tcorr_hdu[0].header)
            
            for key in tfits_hdu[0].header.keys() :
                if key in tcorr_hdu[0].header.keys() :
                    tfits_hdu[0].header[key] = tcorr_hdu[0].header[key]
            for key in tfits_hdu[1].header.keys() :
                if key in tcorr_hdu[0].header.keys() :
                    tfits_hdu[1].header[key] = tcorr_hdu[0].header[key]
            
            if os.path.exists(output_name) :
                os.remove(output_name)
            tfits_hdu.writeto(output_name)
            
            tfits_hdu.close()
            tcorr_hdu.close()
            
            nmatch += 1
            lastj=j
            break

print(nmatch)
