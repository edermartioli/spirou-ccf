# spirou-ccf
Wrapper to perform CCF measurements and CCF analysis to obtain radial velocities from SPIRou data. 

To start using this tool one needs a set of SPIRou spectra in `*t.fits` format saved in the same directory.  Then run the following example:

```
python ~/spirou-ccf/spirou_ccf_pipeline.py --input=*t.fits -vs
```

The following input options are available:
```
--input for input data pattern. 
  (e.g., --pattern=*t.fits)
-s for saving template spectrum to `OBJECT_template.fits`
-p for plotting
-v for verbose
```

NOTE: if one wants the radial velocities corrected by the instrumental drifts measured from the Fabry-Perot "fiber-C" channel, it also requires either the `*o_pp_e2dsff_C_ccf_smart_fp_mask_C.fits` or `e.fits` files saved in the same directory. 

Once the processing is done, one can find the `*_rv.rdb` products, which give the radial velocity time series for several methods used, and the `*_bisector.rdb` file, which gives the bisector time series. One can also find the CCF data for each exposure saved with the following file name structure, `CCFTABLE_FILENAME_MASK-NAME.fits`, as well as the `.csv` table with extended data provided by the CCF analysis. 

If one wants to run the full processing for a given target at LAM server, there is an example script called `run_ccf_at_lam.py` that can be used as reference. Here's an example on how to run it as a background job (will continue if connection is closed):

```
nohup python run_ccf_at_lam.py --workdir="/home/external/emartioli/Data/" > log_of_reduction.out 2>&1 &
```

Note that it requires one to change the internal paths to make this routine useful for a different user location, and also the target list is currenlty provided either by an input list of target names, or if nothing is provided it uses a list defined inside the file, which must be changed. 
