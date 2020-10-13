# spirou-ccf
Wrapper to perform CCF measurements and CCF analysis to obtain radial velocities from SPIRou data. 

To start using this tool one needs a set of SPIRou spectra in `*t.fits` format saved in the same directory.  Then run the following example:

```
python ~/spirou-ccf/spirou_ccf_pipeline.py --input=*t.fits -v
```

The following input options are available:
```
--input for input data pattern. 
  (e.g., --pattern=*t.fits)
-s for saving template spectrum to `OBJECT_template.fits`
-p for plotting
-v for verbose
```

NOTE: if one wants the radial velocities corrected by the instrumental drifts, it also requires either the `*o_pp_e2dsff_C_ccf_smart_fp_mask_C.fits` or `e.fits` files to obtain the drift measurements from the Fabry-Perot "fiber-C" channel. 

Once the processing is done, one can find the `*_rv.rdb` products, which gives the radial velocity time series for several methods used, and the `*_bisector.rdb` file, which gives the bisector time series. One can also find the CCF data for each exposure saved with the following file name structure, `CCFTABLE_FILENAME_MASK-NAME.fits`, as well as the `.csv` table with extended data provided by the CCF analysis. 
