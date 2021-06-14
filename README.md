# spirou-ccf
A tool-kit to perform optimal CCF measurements and a CCF analysis to obtain precise radial velocities from SPIRou data. 

To start using this tool one needs a set of SPIRou spectra in `*t.fits` format saved in the same directory.  Then run the following example:

```
python ~/spirou-ccf/spirou_ccf_pipeline.py --input=*t.fits -pvs
```

The following input options are available:
```
--input for input data pattern. 
  (e.g., --pattern=*t.fits)
--ccf_mask to input a CCF mask (optional, if not provided it will select the best matching mask from a repository)
  (e.g., --ccf_mask=my_favorite_mask.mas)
--source_rv to input the systemic velocity in km/s (if not provided it will measure the RV in the template spectrum)
  (e.g., --source_rv=10.)
--output_template file name for output template (optional, if not provided the template will not be saved)
  (e.g., --output_template='object_name.fits')
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
-d for calculating the instrumental RV drift correction obtained from a drift database (currently not fully implemented)
-p for plotting
-v for verbose
```

Once the processing is done, one can find the `*_rv.rdb` products, which give the radial velocity time series for several methods used, and the `*_bisector.rdb` file, which gives the bisector time series. One can also find the CCF data for each exposure saved with the following file name structure, `CCFTABLE_FILENAME_MASK-NAME.fits`, as well as the `.csv` table with extended data provided by the CCF analysis. 
