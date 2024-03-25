def calculate_bvtv(
    scaling,
    slope,
    intercept,
    BMD_array,
    CORTMASK_array,
    TRABMASK_array,
    cfg,
    IMTYPE: str,
):
    """
    Calculate BVTV and mask images
    ------------------------------
    Scaling, slope and intercept are printed out for review
    If image is already in BMD units, Scaling, Slope and
    Intercept are not applied. BVTVraw is scaled according to
    Hosseini et al. 2017
    (This scaling function could as well be defined externally).

    Parameters
    ----------
    bone    bone results dictionary
    config  configuration parameters dictionary
    IMTYPE  string defining the type of image (BMD/NATIVE)

    Returns
    -------
    bone    bone results dictionary
    """
    print("\n ... prepare mask and BVTV images")
    print("     -> Scaling   = ", scaling)
    print("     -> Slope     = ", slope)
    print("     -> Intercept = ", intercept)

    if IMTYPE.find("BMD") > -1:
        # if image is already in BMD units (e.g. Hosseinis data)
        BVTVraw = BMD_array / 1200.0
    elif IMTYPE.find("NATIVE") > -1:
        BMD_array = (BMD_array / scaling) * slope + intercept
        BVTVraw = BMD_array / 1200.0  # if image is in native units

    # BVTV scaling
    if cfg.image_processing.bvtv_scaling == 1:
        seg_scaling_slope = cfg.image_processing.bvtv_slope
        seg_scaling_intercept = cfg.image_processing.bvtv_intercept
        BVTVscaled = seg_scaling_slope * BVTVraw + seg_scaling_intercept
    else:
        BVTVscaled = BVTVraw

    # set bone values
    MASK = CORTMASK_array + TRABMASK_array
    MASK[MASK > 0] = 1
    BVTVscaled = BVTVscaled * MASK
    BMDscaled = BVTVscaled * 1200 * MASK
    BVTVraw = BVTVraw * MASK
    return BVTVscaled, BMDscaled, BVTVraw
