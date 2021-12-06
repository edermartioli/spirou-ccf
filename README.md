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
--vel_sampling provide a constant sampling in velocity space in km/s (if not provided it will assume a default value of 1.8 km/s)
  (e.g., --vel_sampling=2.0)
--epoch_max_dt provide a maximum time difference to stack spectra in days (if not provided it will assume a default value of 0.5 d)
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
