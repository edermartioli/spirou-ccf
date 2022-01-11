# spirou-ccf
A tool-kit to perform optimal CCF measurements and a CCF analysis to obtain precise radial velocities from SPIRou data. 

To start using this tool one needs a set of SPIRou spectra in `*t.fits` format saved in the same directory.  Then run the following example:

```
python ~/spirou-ccf/spirou_ccf_pipeline.py --input=*t.fits -pv
```

The following input options are available:
```
--input for input data pattern. 
  (e.g., --input=*t.fits)
  
--input_fp_fiberC for FP fiber C E2DS data pattern (optional, necessary only for drift calculation)
  (e.g., --input_fp_fiberC=2??????e.fits or --input_fp_fiberC=2??????o_pp_e2dsff_C.fits)
  
--ccf_mask to input a CCF mask (optional, if not provided it will select the best matching mask from a repository)
  (e.g., --ccf_mask=my_favorite_mask.mas)
  
--output_template file name for output template (optional, if not provided the template will not be saved)
  (e.g., --output_template='object_name.fits')
  
--sci_drift_data to input a .rdb file containing drift data from FP fiber C science frames (optional, necessary only 
for drift calculation, if not provided drifts will be calculated from input_fp_fiberC data, or if no input_fp_fiberC 
provided then drifts will be set to zero)
  (e.g., --sci_drift_data=sci_fiberC_fp_drifts.rdb) 
  
--cal_drift_data to input a .rdb file containing drift data from FP+FP calibrations (optional, necessary only for drift 
calculation, if not provided drifts will be set to zero)
  (e.g., --cal_drift_data=cal_fiberC_fp_drifts.rdb)
  
--source_rv to input the systemic velocity in km/s (if not provided it will measure the RV in the template spectrum)
  (e.g., --source_rv=10.)
  
--ccf_width half width to calculate CCF in km/s (if not provided it will measure the width in the template spectrum)
  (e.g., --ccf_width=100.)
  
--vel_sampling provide a constant sampling in velocity space in km/s (if not provided it will assume a default value 
of 1.8 km/s)
  (e.g., --vel_sampling=2.0)
  
--epoch_max_dt provide a maximum time difference to stack spectra in days (if not provided it will assume a default 
value of 0.5 d)
  (e.g., --epoch_max_dt=0.2)
  
-e for using *e.fits data saved in the same directory (this is useful when calculating telluric CCFs)

-s for stacking the spectrum sequences (polar or per epoch)

-l for stacking a polar sequence rather than a per-epoch sequence

-t to run CCF analysis on telluric spectra. This takes more time to run and do not affect the science CCF calculations.

-d to calcualte the instrumental RV drift correction obtained from simultaneous FP spectra taken through fiber-C and 
nightly calibration FP spectra. The drifts on science exposures can be measured either from *e.fits or o_pp_e2dsff_C.fits 
data saved in the same directory as the input spectra or specified using the option " --input_fp_fiberC", or one can 
input the drift data in a .rdb file using the option "--sci_drift_data". The calibration drifts are obtained by default 
from an existing rdb file saved in drifts/CALIBRATION_FiberC__smart_fp_mask.mas__False__0.6.132__YJHK_fpdrift.rdb or 
one can input the drfit data using the option "--cal_drift_data".

-p for plotting

-v for verbose
```

Once the processing is done, one can find the `*_rv.rdb` products, which give the radial velocity time series for several methods used, and the `*_bisector.rdb` file, which gives the bisector time series. One can also find the CCF data for each exposure saved with the following file name structure, `CCFTABLE_FILENAME_MASK-NAME.fits`, as well as the `.csv` table with extended data provided by the CCF analysis. 



*****************************************
# How to create a CCF drift `".rdb"` file:
*****************************************

## First get all fiber-C spectra from FP-FP calibrations into the same directory, for example:

```
ln -s /net/GSP/nas12c/big_spirou/FULL_REDUCTION_CALIBDB_06131/reduced/2021*/*a_pp_e2dsff_C.fits .
```

or one may copy these files from the server into a local machine:

```
scp -oProxyJump=$USER@charon.osupytheas.fr $USER@cc.lam.fr:/net/GSP/nas12c/big_spirou/FULL_REDUCTION_CALIBDB_06131/reduced/2021*/*a_pp_e2dsff_C.fits .
```

Note in the example above the wildcard `"2021*"` selects only calibrations from 2021, however one may expand or restrict this selection. A good strategy is to include all calibration files obtained in the observing runs that also observed the science target.

## Run FP CCF pipeline:

```
python ~/spirou-ccf/spirou_fp_ccf.py --input=*a_pp_e2dsff_C.fits -pv 
```

## The final CCF drift file should be something like this:

```
CALIBRATION_FiberC__smart_fp_mask.mas__False__0.6.132__YJHK_fpdrift.rdb
```

##The contents of this file is the following:

```
rjd	vrad	svrad	SBCDEN_P	DATE-OBS	WAVETIME	FIBER	EXPTYPE	FILENAME	
---	----	-----	--------	--------	-------	-------	-------
59207.6115577999	-0.96634	0.00357	1.19	2020-12-24	59207.1333517521	C	ALIGN	2571217a
59208.2297991998	-0.96612	0.00340	1.20	2020-12-24	59207.7516637521	C	ALIGN	2571424a
59208.5912780999	-0.96565	0.00327	1.19	2020-12-25	59208.1130737521	C	ALIGN	2571454a
59209.2326638000	-0.96587	0.00380	1.19	2020-12-25	59208.7544565521	C	ALIGN	2571641a
59209.5913188001	-0.96537	0.00336	1.19	2020-12-26	59209.1131080521	C	ALIGN	2571671a
...
```

Note that the same procedure above can be done in the science fiber-C data or `*e.fits` files to obtain similar drift files for the simultaneous FP exposures.  These `*.rdb` files are inputs to the `spirou_ccf_pipeline.py` so you can have more control on the drifts. 
