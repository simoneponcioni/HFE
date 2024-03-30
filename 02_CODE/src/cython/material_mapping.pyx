cimport numpy as np
import numpy as np
from libc.math cimport abs
import logging
import warnings

cdef logger = logging.getLogger(__name__)

cdef __compute_roi_indices__(double x, double y, double z, double VOI, int[:] seg_array_shape):
    cdef:
        int x_start, x_end, y_start, y_end, z_start, z_end
        double x_voi_half = x - VOI / 2
        double y_voi_half = y - VOI / 2
        double z_voi_half = z - VOI / 2

    x_start, x_end = int(round(max(x_voi_half, 0))), int(
        round(min(x_voi_half + VOI, seg_array_shape[0]))
    )
    y_start, y_end = int(round(max(y_voi_half, 0))), int(
        round(min(y_voi_half + VOI, seg_array_shape[1]))
    )
    z_start, z_end = int(round(max(z_voi_half, 0))), int(
        round(min(z_voi_half + VOI, seg_array_shape[2]))
    )
    return x_start, x_end, y_start, y_end, z_start, z_end

cdef __computePHI__(np.ndarray[np.uint8_t, ndim=1] ROI):
    cdef double phi
    try:
        phi = float(ROI.sum()) / ROI.size
    except ZeroDivisionError:
        phi = 0.0

    if phi != phi:  # equivalent to np.isnan(phi)
        phi = 0.0
    if phi > 1:
        phi = 1.0
    return phi

cdef __generate_sphere_array__(shape, radius, position):
    # You need to provide the implementation for this function
    pass

cdef __compute_bvtv__(double x, double y, double z, double VOI, np.ndarray[np.uint8_t, ndim=1] seg_array, np.ndarray[np.uint8_t, ndim=1] mask_array):
    cdef:
        list seg_array_shape = [seg_array.shape[i] for i in range(seg_array.ndim)]
        int x_start, x_end, y_start, y_end, z_start, z_end
        double phi_s, rho_s
        np.ndarray[np.uint8_t, ndim=1] ROI_seg, ROI_mask, ROI_mask_sphere, ROI_mask_sphere_mask, ROI_sphere_seg

    x_start, x_end, y_start, y_end, z_start, z_end = __compute_roi_indices__(x, y, z, VOI, seg_array_shape)

    ROI_seg = seg_array[x_start:x_end, y_start:y_end, z_start:z_end]
    ROI_mask = mask_array[x_start:x_end, y_start:y_end, z_start:z_end]

    phi_s = __computePHI__(ROI_mask)

    if phi_s > 0.0:
        ROI_mask_sphere = __generate_sphere_array__(
            shape=[ROI_seg.shape[i] for i in range(ROI_seg.ndim)],
            radius=VOI / 2,
            position=[
                cog_i - start_i
                for cog_i, start_i in zip([x, y, z], [x_start, y_start, z_start])
            ],
        )

        ROI_mask_sphere_mask = ROI_mask_sphere * ROI_mask
        ROI_sphere_seg = ROI_mask_sphere_mask * ROI_seg
        with warnings.catch_warnings():
            warnings.filterwarnings("error")
            try:
                rho_s = ROI_sphere_seg.sum() / ROI_mask_sphere_mask.sum()
            except Warning as e:
                rho_s = 0.01

    else:
        phi_s = 0.01
        rho_s = 0.01
    return max(0.01, min(1.0, phi_s)), max(0.01, min(1.0, rho_s))

cpdef np.ndarray[np.float64_t, ndim=1] calculate_bvtv(
    np.ndarray[np.uint8_t, ndim=1] seg_array,
    np.ndarray[np.uint8_t, ndim=1] mask_array,
    np.ndarray[np.float64_t, ndim=1] spacing,
    np.float64_t VOI_mm,
    np.ndarray[np.float64_t, ndim=1] cog,
):
    cdef:
        double VOI = VOI_mm / spacing[0]
        double x = cog[0] / spacing[0]
        double y = cog[1] / spacing[1]
        double z = cog[2] / spacing[2]

    return __compute_bvtv__(x, y, z, VOI, seg_array, mask_array)

cpdef np.ndarray[np.float64_t, ndim=1] material_mapping(
    np.ndarray[np.float64_t, ndim=1] cog,
    np.ndarray[np.float64_t, ndim=1] spacing,
    np.float64_t VOI_mm,
    np.ndarray[np.uint8_t, ndim=1] mask_array,
    np.ndarray[np.uint8_t, ndim=1] seg_array,
    bint SEG_correction,
    bint all_mask,
    str compartment_s,
):
    cdef:
        np.ndarray[np.float_t, ndim=1] phi = np.empty(len(cog), dtype=np.float)
        np.ndarray[np.float_t, ndim=1] rho = np.empty(len(cog), dtype=np.float)
        np.ndarray[np.float_t, ndim=1] rho_fe = np.empty(len(cog), dtype=np.float)
        int i
        double phi_s, rho_s, TOL_SPACING = 0.01

    for i in range(seg_array.size):
        seg_array[i] = 1 if seg_array[i] > 0.1 else 0
    for i in range(mask_array.size):
        mask_array[i] = 1 if mask_array[i] > 0.1 else 0

    if not SEG_correction:
        logger.debug("SEG_correction is set to False")

    for i, cog_s in enumerate(cog):

        phi_s, rho_s = calculate_bvtv(seg_array, mask_array, spacing, VOI_mm, cog_s)

        if SEG_correction:
            if abs(spacing[0] - 0.061) <= TOL_SPACING:
                rho_s = rho_s * 0.651 + 0.056462
            elif abs(spacing[0] - 0.082) <= TOL_SPACING:
                rho_s = rho_s * 0.745745 - 0.0209902
            else:
                raise ValueError(
                    f"SEG_correction is True but 'spacing' is not 0.061 nor 0.082 (it is {spacing[0]}))"
                )

        if all_mask and phi_s > 0.0 and rho_s < 0.01:
            rho_s = 0.01

        phi[i] = phi_s
        rho[i] = rho_s
        rho_fe[i] = 1  # ! adapt this
    return phi, rho, rho_fe