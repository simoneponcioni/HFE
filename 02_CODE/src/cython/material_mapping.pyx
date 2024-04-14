import numpy as np
cimport numpy as cnp
cimport cython
import warnings
cnp.import_array()

DTYPE = np.float64
ctypedef cnp.int64_t DTYPE_t


cdef cnp.ndarray[DTYPE_t, ndim=1] threshold_seg_array(cnp.ndarray[DTYPE_t, ndim=1] seg_array):
    seg_array[seg_array > 0.1] = 1
    seg_array[seg_array <= 0.1] = 0
    return seg_array


cdef float __segmentation_calibration__(float rho_s, spacing):
    """
    Calibration of the segmentation threshold
    Simone Poncioni, MSB, 04.2024
    """
    cdef TOL_SPACING = 0.01
    cdef float spacing_0 = spacing[0]

    if abs(spacing_0 - 0.061) <= TOL_SPACING:
        # Correction curve from Varga et al. 2009 for XCTII
        rho_s = rho_s * 0.651 + 0.056462
    elif abs(spacing_0 - 0.082) <= TOL_SPACING:
        # Correction curve from Varga et al. 2009 for XCTI, added by MI
        rho_s = rho_s * 0.745745 - 0.0209902
    else:
        raise ValueError(
            f"SEG_correction is True but 'spacing' is not 0.061 nor 0.082 (it is {spacing_0})"
        )
    return rho_s


cdef tuple[int, int, int, int, int, int] __compute_roi_indices__(float x, float y, float z, int VOI, int[:,:,:] seg_array):
    """
    Compute the start and end indices for a region of interest (ROI) in a 3D array.

    The ROI is centered at a point (x, y, z) and has a size of VOI (Volume of Interest) in each dimension.
    The function ensures that the indices do not go beyond the array boundaries.

    Parameters:
    x (int): The x-coordinate of the center of the ROI.
    y (int): The y-coordinate of the center of the ROI.
    z (int): The z-coordinate of the center of the ROI.
    VOI (int): The size of the Volume of Interest in each dimension.
    seg_array (np.array): The 3D array in which to calculate the ROI.

    Returns:
    tuple: A tuple containing the start and end indices for the x, y, and z dimensions.
    """
    cdef float x_start, x_end, y_start, y_end, z_start, z_end
    x_start, x_end = int(np.rint(max(x - VOI / 2, 0))), int(
        np.rint(min(x + VOI / 2, seg_array.shape[0]))
    )
    y_start, y_end = int(np.rint(max(y - VOI / 2, 0))), int(
        np.rint(min(y + VOI / 2, seg_array.shape[1]))
    )
    z_start, z_end = int(np.rint(max(z - VOI / 2, 0))), int(
        np.rint(min(z + VOI / 2, seg_array.shape[2]))
    )
    return x_start, x_end, y_start, y_end, z_start, z_end


cdef float __computePHI__(int[:,:,:] ROI):
    """
    Compute the bone partial volume.

    Parameters:
    ROI (ndarray): The array containing the ROI of the image.

    Returns:
    PHI (float): The PHI value.
    """
    cdef float phi
    cdef int ROI_size = ROI.shape[0] * ROI.shape[1] * ROI.shape[2]
    try:
        phi = <float>np.count_nonzero(ROI) / ROI_size
    except ZeroDivisionError:
        phi = 0.0

    # check for meaningful output
    if np.isnan(phi):
        phi = 0.0
    if phi > 1:
        phi = 1.0
    return phi


cdef __generate_sphere_array__(shape, radius, position):
    """
    Generate a 3D array with a sphere of given radius and position.

    The function creates a 3D grid of the specified shape, and then calculates for each point in the grid
    if it's inside the sphere (represented as 1) or outside the sphere (represented as 0).

    Parameters:
    shape (tuple): The shape of the 3D array to be generated.
    radius (float): The radius of the sphere.
    position (tuple): The position of the center of the sphere.

    Returns:
    np.array: A 3D numpy array of the specified shape.
    """
    semisizes = (float(radius),) * 3
    grid = [slice(-x0, dim - x0) for x0, dim in zip(position, shape)]
    position = np.ogrid[grid]
    arr = np.zeros(np.asarray(shape).astype(int), dtype=float)
    for x_i, semisize in zip(position, semisizes):
        arr += np.abs(x_i / semisize) ** 2
    return (arr <= 1.0).astype("int")


def __compute_bvtv__python(seg_array, mask_array, x, y, z, VOI):
    """
    Compute the bone volume to total volume (BV/TV) ratio.

    This function computes the BV/TV ratio by first generating a sphere array within the ROI,
    then calculating the PHI and rho_s values. If the PHI value is greater than 0,
    the function calculates the rho_s value. If the PHI value is not greater than 0,
    the function sets both the PHI and rho_s values to 0.01.

    Returns:
    tuple: A tuple containing the PHI and rho_s values.
    """
    cdef float phi_s, rho_s
    x_start, x_end, y_start, y_end, z_start, z_end = __compute_roi_indices__(
        x, y, z, VOI, seg_array
    )

    ROI_seg = seg_array[x_start:x_end, y_start:y_end, z_start:z_end]
    ROI_mask = mask_array[x_start:x_end, y_start:y_end, z_start:z_end]

    phi_s = __computePHI__(ROI_mask)

    if phi_s > 0.0:
        ROI_mask_sphere = __generate_sphere_array__(
            shape=np.shape(ROI_seg),
            radius=VOI / 2,
            position=[
                cog_i - start_i
                for cog_i, start_i in zip([x, y, z], [x_start, y_start, z_start])
            ],
        )

        ROI_mask_sphere_mask = np.multiply(ROI_mask_sphere, ROI_mask)
        ROI_sphere_seg = np.multiply(ROI_mask_sphere_mask, ROI_seg)
        with warnings.catch_warnings():
            warnings.filterwarnings("error")
            try:
                rho_s = np.sum(ROI_sphere_seg) / np.sum(ROI_mask_sphere_mask)
            except Warning as e:
                rho_s = 0.01

    else:
        # Ensure minimum bvtv, added by POS, 14.01.2024
        phi_s = 0.01
        rho_s = 0.01
    return max(0.01, min(1.0, phi_s)), max(0.01, min(1.0, rho_s))


cdef tuple[float, float] __compute_bvtv__(int[:, :, :] seg_array, int[:, :, :] mask_array, float x, float y, float z, int VOI):
    """
    Compute the bone volume to total volume (BV/TV) ratio.

    This function computes the BV/TV ratio by first generating a sphere array within the ROI,
    then calculating the PHI and rho_s values. If the PHI value is greater than 0,
    the function calculates the rho_s value. If the PHI value is not greater than 0,
    the function sets both the PHI and rho_s values to 0.01.

    Returns:
    tuple: A tuple containing the PHI and rho_s values.
    """
    cdef float phi_s, rho_s
    cdef int x_start, x_end, y_start, y_end, z_start, z_end
    cdef int[:,:,:] ROI_seg, ROI_mask, seg_array_view, mask_array_view
    x_start, x_end, y_start, y_end, z_start, z_end = __compute_roi_indices__(
        x, y, z, VOI, seg_array_view
    )

    ROI_seg = seg_array[x_start:x_end, :, :]
    ROI_mask = mask_array[x_start:x_end, y_start:y_end, z_start:z_end]

    phi_s = __computePHI__(ROI_mask)

    if phi_s > 0.0:
        cdef int[:,:,:] ROI_mask_sphere = __generate_sphere_array__(
            shape=(x_end-x_start, y_end-y_start, z_end-z_start),
            radius=VOI / 2,
            position=[
                cog_i - start_i
                for cog_i, start_i in zip([x, y, z], [x_start, y_start, z_start])
            ],
        )

        cdef int[:,:,:] ROI_mask_sphere_mask = ROI_mask_sphere * ROI_mask
        cdef int[:,:,:] ROI_sphere_seg = ROI_mask_sphere_mask * ROI_seg
        try:
            rho_s = np.sum(ROI_sphere_seg) / np.sum(ROI_mask_sphere_mask)
        except ZeroDivisionError:
            rho_s = 0.01

    else:
        # Ensure minimum bvtv, added by POS, 14.01.2024
        phi_s = 0.01
        rho_s = 0.01
    return max(0.01, min(1.0, phi_s)), max(0.01, min(1.0, rho_s))


def calculate_bvtv(seg_array, mask_array, spacing, VOI_mm, cog):
    """
    POS, 06.02.2024
    Calculate the bone volume to total volume (BV/TV) ratio.

    This function calculates the BV/TV ratio, which is a measure of bone density.
    It does this by first calculating the region of interest (ROI) in the given 3D array (seg_array),
    then computing the bone partial volume (PHI), and finally computing the BV/TV ratio.

    Parameters:
    seg_array (np.array): The 3D array representing the segmented image.
    mask_array (np.array): The 3D array representing the mask image.
    spacing (np.array): The spacing between voxels in the 3D image.
    VOI_mm (float): The size of the Volume of Interest in millimeters.
    cog (np.array): The center of gravity of the FE element.

    Returns:
    tuple: A tuple containing the PHI and RHO values.
    """
    cdef float VOI, x, y, z, phi, rho

    VOI = VOI_mm / spacing[0]
    x, y, z = cog / spacing
    phi, rho = __compute_bvtv__(seg_array, mask_array, x, y, z, VOI)

    return phi, rho


def __material_mapping__(
    cog,
    spacing,
    VOI_mm,
    mask_array,
    seg_array,
    SEG_correction,
    all_mask,
):
    """
    Compartment agnostic material mapping function
    Simone Poncioni, MSB, 08.2023
    """
    len_cog = len(cog)
    phi = np.zeros(len_cog)
    rho = np.zeros(len_cog)
    rho_fe = np.ones(len_cog)

    for i, cog_s in enumerate(cog):
        phi_s, rho_s = calculate_bvtv(seg_array, mask_array, spacing, VOI_mm, cog_s)

        if SEG_correction == True:
            rho_s = __segmentation_calibration__(rho_s, spacing)

        if all_mask == True:
            if phi_s > 0.0 and rho_s < 0.01:
                rho_s = 0.01

        phi[i] = phi_s
        rho[i] = rho_s
        # rho_fe[i] = 1  # ! adapt this

    return phi, rho, rho_fe


def exec_mapping(
    cnp.ndarray[DTYPE_t, ndim=1] cog,
    float[:] spacing,
    cnp.double_t VOI_mm,
    cnp.ndarray[DTYPE_t, ndim=1] mask_array,
    cnp.ndarray[DTYPE_t, ndim=1] seg_array,
    bool SEG_correction,
    bool all_mask,
):

    seg_array = threshold_seg_array(seg_array)
    mask_array = threshold_seg_array(mask_array)

    phi, rho, rho_fe = __material_mapping__(
        cog,
        spacing,
        VOI_mm,
        mask_array,
        seg_array,
        SEG_correction,
        all_mask,
    )

    return phi, rho, rho_fe