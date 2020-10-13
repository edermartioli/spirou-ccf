# spirou-ccf
Wrapper to perform CCF measurements and CCF analysis to obtain radial velocities from SPIRou data. 

To start using this tool one needs a set of SPIRou spectra in `*t.fits` format saved in the same directory.  Then run the following example:

```
python ~/spirou-ccf/spirou_ccf_pipeline.py --input=*t.fits -v
```

NOTE: if one wants the radial velocities corrected by the instrumental drifts, it also requires either the `*o_pp_e2dsff_C_ccf_smart_fp_mask_C.fits` or `e.fits` files to obtain the drift measurements from the Fabry-Perot "fiber-C" channel. 

