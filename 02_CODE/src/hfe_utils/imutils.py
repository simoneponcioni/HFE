import logging
from pathlib import Path
from struct import unpack

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import SimpleITK as sitk
import vtk  # type: ignore
from hfe_utils.io_utils import ext
from mpl_toolkits.axes_grid1 import make_axes_locatable  # type: ignore
from vtk.util.numpy_support import numpy_to_vtk, vtk_to_numpy  # type: ignore

# flake8: noqa: E501


LOGGING_NAME = "HFE-ACCURATE"
logger = logging.getLogger(LOGGING_NAME)
logger.propagate = False


def vtk2numpy(imvtk):
    """turns a vtk image data into a numpy array"""
    dim = imvtk.GetDimensions()
    data = imvtk.GetPointData().GetScalars()
    imnp = vtk_to_numpy(data)
    # vtk and numpy have different array conventions
    imnp = imnp.reshape(dim[2], dim[1], dim[0])
    imnp = imnp.transpose(2, 1, 0)
    return imnp


def numpy2vtk(imnp, spacing):
    """turns a numpy array into a vtk image data"""
    # vtk and numpy have different array conventions
    imnp_flat = imnp.transpose(2, 1, 0).flatten()
    if imnp.dtype == "int8":
        arraytype = vtk.VTK_CHAR
    elif imnp.dtype == "int16":
        arraytype = vtk.VTK_SHORT
    else:
        arraytype = vtk.VTK_FLOAT
    imvtk = numpy_to_vtk(num_array=imnp_flat, deep=True, array_type=arraytype)
    image = vtk.vtkImageData()
    image.SetDimensions(imnp.shape)
    image.SetSpacing(spacing)
    points = image.GetPointData()
    points.SetScalars(imvtk)
    return image


def pad_image(image, iso_pad_size: int):
    """
    Pads the input image with a constant value (background value) to
    increase its size.
    Padding is used to prevent having contours on the edges of the image,
    which would cause the spline fitting to fail.
    Padding is performed on the transverse plane only
    (image orientation is assumed to be z, y, x)

    Args:
        image (SimpleITK.Image): The input image to be padded.
        iso_pad_size (int): The size of the padding to be added
                            to each dimension.

    Returns:
        SimpleITK.Image: The padded image.
    """
    constant = int(sitk.GetArrayFromImage(image).min())
    image_pad = sitk.ConstantPad(
        image,
        (0, iso_pad_size, iso_pad_size),
        (0, iso_pad_size, iso_pad_size),
        constant,
    )
    return image_pad


def __save_image_with_colorbar__(data, output_path):
    plt.figure()
    ax = plt.gca()
    im = ax.imshow(data, cmap="viridis")
    plt.title(
        f"{Path(output_path).parent.name} - {Path(output_path).stem}",
        fontsize=20,
        weight="bold",
    )
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(im, cax=cax)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.clf()
    plt.close()
    return None


def save_images_with_colorbar(cfg, sample, bone):
    """
    Saves images of arrays with or without colorbar.

    Args:
        config (dict): Configuration dictionary containing paths and other parameters.
        sample (str): The sample identifier.
        bone (dict): Dictionary containing arrays to be saved as images.

    The function loops over the arrays in the bone dictionary and saves images.
    If the array is "BMD_array" or "SEG_array", it uses the provided function
    to save the image with a colorbar. For other arrays, it saves the image without a colorbar.
    """
    # Write out test images to check quality
    impath = Path(cfg.paths.aimdir, cfg.simulations.folder_id[sample], "")
    Path(impath).mkdir(parents=True, exist_ok=True)

    # Loop over the arrays and save images with colorbar
    for array_name in [
        "BMD_array",
        "SEG_array",
        "TRABMASK_array",
        "CORTMASK_array",
    ]:
        array_data = bone[array_name][int(bone[array_name].shape[0] / 2), :, :]
        # transpose the array to match the image orientation
        array_data = np.transpose(array_data, (1, 0))
        image_path = str(Path(impath / (array_name.upper().split("_")[0] + ".png")))

        # If the array is BMD_array or SEG_array, use the function that handles colorbar
        if array_name == "BMD_array" or array_name == "SEG_array":
            __save_image_with_colorbar__(array_data, image_path)
        else:
            # For other arrays, save without colorbar
            mpimg.imsave(image_path, array_data)
    return None


def get_AIM_ints(f):
    """Function by Glen L. Niebur, University of Notre Dame (2010)
    reads the integer data of an AIM file to find its header length"""
    nheaderints = 32
    f.seek(0)
    binints = f.read(nheaderints * 4)
    header_int = unpack("=32i", binints)
    return header_int


def AIMreader(fileINname, spacing):
    """reads an AIM file and provides the corresponding vtk image
    with spacing, calibration data and header"""
    # read header
    print("     " + fileINname)
    with open(fileINname, "rb") as f:
        AIM_ints = get_AIM_ints(f)
        # check AIM version
        if int(AIM_ints[5]) == 16:
            print("     -> version 020")
            if int(AIM_ints[10]) == 131074:
                format = "short"
                print("     -> format " + format)
            elif int(AIM_ints[10]) == 65537:
                format = "char"
                print("     -> format " + format)
            elif int(AIM_ints[10]) == 1376257:
                format = "bin compressed"
                print("     -> format " + format + " not supported! Exiting!")
                exit(1)
            else:
                format = "unknown"
                print("     -> format " + format + "! Exiting!")
                exit(1)
            header = f.read(AIM_ints[2])
            header_len = len(header) + 160
            extents = (0, AIM_ints[14] - 1, 0, AIM_ints[15] - 1, 0, AIM_ints[16] - 1)
        else:
            print("     -> version 030")
            if int(AIM_ints[17]) == 131074:
                format = "short"
                print("     -> format " + format)
            elif int(AIM_ints[17]) == 65537:
                format = "char"
                print("     -> format " + format)
            elif int(AIM_ints[17]) == 1376257:
                format = "bin compressed"
                print("     -> format " + format + " not supported! Exiting!")
                exit(1)
            else:
                format = "unknown"
                print("     -> format " + format + "! Exiting!")
                exit(1)
            header = f.read(AIM_ints[8])
            header_len = len(header) + 280
            extents = (0, AIM_ints[24] - 1, 0, AIM_ints[26] - 1, 0, AIM_ints[28] - 1)

    # collect data from header if existing
    # header = re.sub('(?i) +', ' ', header)
    header = header.split("\n".encode())
    header.pop(0)
    header.pop(0)
    header.pop(0)
    header.pop(0)
    scaling = None
    slope = None
    intercept = None
    IPLPostScanScaling = 1
    for line in header:
        if line.find(b"Orig-ISQ-Dim-p") > -1:
            origdimp = [int(s) for s in line.split(b" ") if s.isdigit()]

        if line.find("Orig-ISQ-Dim-um".encode()) > -1:
            origdimum = [int(s) for s in line.split(b" ") if s.isdigit()]

        if line.find("Orig-GOBJ-Dim-p".encode()) > -1:
            origdimp = [int(s) for s in line.split(b" ") if s.isdigit()]

        if line.find("Orig-GOBJ-Dim-um".encode()) > -1:
            origdimum = [int(s) for s in line.split(b" ") if s.isdigit()]

        if line.find("Scaled by factor".encode()) > -1:
            scaling = float(line.split(" ".encode())[-1])
        if line.find("Density: intercept".encode()) > -1:
            intercept = float(line.split(" ".encode())[-1])
        if line.find("Density: slope".encode()) > -1:
            slope = float(line.split(" ".encode())[-1])
            # if el_size scale was applied, the above still takes the original
            # voxel size. This function works only if an isotropic scaling
            # is applied!
        if line.find("downscaled".encode()) > -1:
            # avoids that the filename 'downscaled' is interpreted as
            # a scaling factor
            pass
        elif line.find("scale".encode()) > -1:
            IPLPostScanScaling = float(line.split(" ".encode())[-1])
    # Spacing is calculated from Original Dimensions. This is wrong, when
    # the images were coarsened and the voxel size is not anymore
    # corresponding to the original scanning resolution!

    try:
        spacing = IPLPostScanScaling * (
            np.around(np.asarray(origdimum) / np.asarray(origdimp) / 1000, 5)
        )
    except UnboundLocalError:
        pass

    # read AIM
    reader = vtk.vtkImageReader2()
    reader.SetFileName(fileINname)
    reader.SetDataByteOrderToLittleEndian()
    reader.SetFileDimensionality(3)
    reader.SetDataExtent(extents)
    reader.SetHeaderSize(header_len)
    if format == "short":
        reader.SetDataScalarTypeToShort()
    elif format == "char":
        reader.SetDataScalarTypeToChar()
    reader.SetDataSpacing(spacing)
    reader.Update()
    imvtk = reader.GetOutput()
    return imvtk, spacing, scaling, slope, intercept, header


def read_img_param(filenames):
    """
    Read image pararmeters from AIM image header.
    Input: AIM image (Scanco Medical)
    Output: (bone dictionary)
    - Spacing
    - Scaling
    - Slope
    - Intercept
    """
    print("\n ... read AIM files")

    """
    read image informations from raw image
    (only for files processed in medtool
    """
    try:
        _, spacing, scaling, slope, intercept, _ = AIMreader(
            filenames["RAWname"], np.array([0.0606997, 0.0606997, 0.0606997])
        )
    except Exception as e:
        logger.exception(f"An error occurred while using AIMreader: {e}")

    return spacing, scaling, slope, intercept


def read_aim(name, filenames, bone, lock):
    """
    read AIM image
    --------------
    All necessary AIM files are imported and stored in bone dict
    Input: name specifier, filenames dict, bone dict
    Output: bone dict
    - numpy array containing AIM image
    """

    print("\n ... read file: " + name)

    spacing = bone["Spacing"]
    IMG_vtk = AIMreader(filenames[name + "name"], spacing)[0]
    IMG_array = vtk2numpy(IMG_vtk)

    # pad image to avoid having non-zero values at the boundary
    IMG_sitk = sitk.GetImageFromArray(IMG_array)
    IMG_pad = pad_image(IMG_sitk, iso_pad_size=10)
    IMG_pad = sitk.Flip(IMG_pad, [True, False, False])

    #! ONLY FOR C0003114
    if "C0003114" in filenames[name + "name"]:
        print('Removing 35 slices')
        IMG_pad = IMG_pad[:-35, :, :]
        print(IMG_pad.GetSize())
    else:
        pass

    IMG_array = sitk.GetArrayFromImage(IMG_pad)
    IMG_array = np.flip(IMG_array, 1)
    if name == "SEG":
        IMG_array[IMG_array == 127] = 2
        IMG_array[IMG_array == 126] = 1
        with lock:
            bone[name + "_array"] = IMG_array
    else:
        with lock:
            bone[name + "_array"] = IMG_array

    print(f"{name} shape: {IMG_array.shape}")
    return bone


class CropType:
    expand = 0
    crop = 1
    variable = 2


def __adjust__img_size__(image, coarsefactor, crop_z=1):
    """
    Images are adjusted according to CropType:
    0 = CropType.expand     (Expand image by copying layers)
    1 = CropType.crop       (Crop image)
    2 = CropType.variable   (Either crop or expand, depending on what includes less layers)
    """

    # measure image shape
    IMDimX = np.shape(image)[0]
    IMDimY = np.shape(image)[1]
    IMDimZ = np.shape(image)[2]

    AddDimX = coarsefactor - (IMDimX % coarsefactor)
    AddDimY = coarsefactor - (IMDimY % coarsefactor)

    # adjust in x and y direction
    shape_diff = [AddDimX, AddDimY]
    xy_image_adjusted = np.lib.pad(
        image,
        ((0, shape_diff[0]), (0, shape_diff[1]), (0, 0)),
        "constant",
        constant_values=(0),
    )

    if crop_z == CropType.crop:
        image_adjusted = xy_image_adjusted

    if crop_z == CropType.expand:
        AddDimZ = coarsefactor - (IMDimZ % coarsefactor)
        shape_diff = [AddDimX, AddDimY, AddDimZ]
        image_adjusted = np.lib.pad(
            xy_image_adjusted, ((0, 0), (0, 0), (0, shape_diff[2])), "edge"
        )

    if crop_z == CropType.variable:
        limit = coarsefactor / 2.0
        if IMDimZ % coarsefactor > limit:
            AddDimZ = coarsefactor - (IMDimZ % coarsefactor)
            shape_diff = [AddDimX, AddDimY, AddDimZ]
            image_adjusted = np.lib.pad(
                xy_image_adjusted, ((0, 0), (0, 0), (0, shape_diff[2])), "edge"
            )
        if IMDimZ % coarsefactor < limit:
            image_adjusted = xy_image_adjusted

    return image_adjusted


def adjust_image_size(name, bone, cfg, croptype=CropType.crop):
    """
    Adjust image size to current FEelement size,
    that no layers of the image are removed, size in z direction has to fit.
    If empty layers are added in this dimension,
    this will create a weak layer at the bottom of the image.
    Expansions in x and y dimension,
    will probably not affect strength, but will lower stiffness.
    """
    # get bone values
    img_array = bone[name + "_array"]
    spacing = bone["Spacing"]

    # coarsening factor = FE element size / CT voxel size
    coarse_factor = int(round(cfg.mesher.element_size / spacing[0]))
    fe_el_size = np.copy(spacing) * coarse_factor

    # Adjustment for BMD image and Mask
    img_array_adjusted = __adjust__img_size__(img_array, coarse_factor, CropType.crop)

    # for XCTI added by MI
    if spacing[0] == 0.082:
        height = img_array.shape[2] * spacing[0]
        elem_n = np.rint(height / cfg.mesher.element_size).astype(int)
        coarse_factor = img_array.shape[2] / elem_n
        fe_el_size = np.copy(spacing) * coarse_factor

    # set bone values
    # copy old IMG_array to IMG_array_original and store new adjusted IMG as IMG_array
    bone[name + "_array_original"] = img_array
    bone[name + "_array"] = img_array_adjusted
    bone["FEelSize"] = fe_el_size
    bone["CoarseFactor"] = coarse_factor

    return bone


def compute_bvtv_d_seg(bone: dict, sample: str) -> dict:
    """
    Compute BVTV from segmented images and from corrected BVTVd values for comparison
    Parameters
    ----------
    bone
    sample

    Returns
    -------
    bone: dict
    """
    logger.info("\n ... compute BVTV and BVTVd")
    SEG = bone["SEG_array"]
    SEG[SEG > 0] = 1
    seg_voxels = np.sum(SEG[SEG > 0])
    MASK = bone["CORTMASK_array"] + bone["TRABMASK_array"]
    MASK[MASK > 0] = 1
    mask_voxels = np.sum(MASK[MASK > 0])
    bone["BVTV_seg_compare"] = seg_voxels / mask_voxels

    BVTVd_scaled = bone["BVTVscaled"]
    mean_BVTVd_corrected = np.mean(BVTVd_scaled[BVTVd_scaled != 0])
    BVTVd_raw = bone["BVTVraw"]
    mean_BVTVd_raw = np.mean(BVTVd_raw[BVTVd_raw != 0])

    bone["mean_BVTV_seg"] = seg_voxels / mask_voxels
    bone["mean_BVTVd_scaled"] = mean_BVTVd_corrected
    bone["mean_BVTVd_raw"] = mean_BVTVd_raw

    return bone


def fab2vtk_fromdict(filename, abq_dict):
    # write vtk files for each eigenvector
    for i in [0, 1, 2]:
        if i == 0:
            vtkname = ext(filename, "_FABmin.vtk")
        elif i == 1:
            vtkname = ext(filename, "_FABmid.vtk")
        elif i == 2:
            vtkname = ext(filename, "_FABmax.vtk")

        with open(vtkname, "w") as vtkFile:
            vtkFile.write("# vtk DataFile Version 2.0\n")
            vtkFile.write("Reconstructed Lagrangian Field Data\n")
            vtkFile.write("ASCII\n")
            vtkFile.write("DATASET UNSTRUCTURED_GRID\n")
            vtkFile.write("\nPOINTS " + str(2 * len(abq_dict.keys())) + " float\n")
            for elem in abq_dict.keys():
                centroid = abq_dict[elem]["centroid"]
                m = abq_dict[elem]["m"]
                mm = abq_dict[elem]["mm"]

                vtkFile.write(
                    str(centroid[0] - m[i] * mm[0][i])
                    + " "
                    + str(centroid[1] - m[i] * mm[1][i])
                    + " "
                    + str(centroid[2] - m[i] * mm[2][i])
                    + "\n"
                )
            for elem in abq_dict.keys():
                centroid = abq_dict[elem]["centroid"]
                m = abq_dict[elem]["m"]
                mm = abq_dict[elem]["mm"]
                vtkFile.write(
                    str(centroid[0] + m[i] * mm[0][i])
                    + " "
                    + str(centroid[1] + m[i] * mm[1][0])  # ? was 0
                    + " "
                    + str(centroid[2] + m[i] * mm[2][i])
                    + "\n"
                )
            vtkFile.write(
                "\nCELLS "
                + str(len(abq_dict.keys()))
                + " "
                + str(3 * len(abq_dict.keys()))
                + "\n"
            )
            count = 0
            for elem in abq_dict.keys():
                vtkFile.write(
                    "2 " + str(count) + " " + str(count + len(abq_dict.keys())) + "\n"
                )
                count += +1
            vtkFile.write("\nCELL_TYPES " + str(len(abq_dict.keys())) + "\n")
            for elem in abq_dict.keys():
                vtkFile.write("3\n")

            vtkFile.write("\nCELL_DATA " + str(len(abq_dict.keys())) + "\n")
            vtkFile.write("scalars DOA_max float\n")
            vtkFile.write("LOOKUP_TABLE default\n")

            for elem in abq_dict.keys():
                m = abq_dict[elem]["m"]
                vtkFile.write(str(m[0] / m[2]) + "\n")

            try:
                vtkFile.write("scalars PHIc float\n")
                vtkFile.write("LOOKUP_TABLE default\n")
                for elem in abq_dict.keys():
                    PHIc = abq_dict[elem]["PHIc"]
                    vtkFile.write(str(PHIc) + "\n")

                vtkFile.write("scalars PHIt float\n")
                vtkFile.write("LOOKUP_TABLE default\n")
                for elem in abq_dict.keys():
                    PHIt = abq_dict[elem]["PHIt"]
                    vtkFile.write(str(PHIt) + "\n")

            except KeyError:
                vtkFile.write("scalars PHI float\n")
                vtkFile.write("LOOKUP_TABLE default\n")
                for elem in abq_dict.keys():
                    PHI = abq_dict[elem]["PHI"]
                    vtkFile.write(str(PHI) + "\n")

            logger.info(" ... vtk file written: " + vtkname)
    return None


def quiver_3d_MSL(eval, evect, cogs, path):
    cmap = "viridis"
    if eval.ptp() != 0:
        c = (eval - eval.min()) / eval.ptp()
    else:
        c = np.zeros_like(eval)  # or some other appropriate value
    c = np.concatenate((c, np.repeat(c, 2)))
    c = getattr(plt.cm, cmap)(c)

    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection="3d")
    q = ax.quiver(
        cogs[:, 0],
        cogs[:, 1],
        cogs[:, 2],
        evect[:, 0],
        evect[:, 1],
        evect[:, 2],
        cmap=cmap,
    )
    fig.colorbar(q, shrink=0.5)
    q.set_edgecolor(c)
    q.set_facecolor(c)
    plt.savefig(path, dpi=300)
    plt.close(fig)


def plot_MSL_fabric_spline(cfg, abq_dict: dict, sample):
    n_elems = len(abq_dict.keys())
    cogs_plot = np.zeros((n_elems, 3))
    evect1 = np.zeros((n_elems, 3))
    evect2 = np.zeros((n_elems, 3))
    evect3 = np.zeros((n_elems, 3))
    eval1 = np.zeros(n_elems)
    eval2 = np.zeros(n_elems)
    eval3 = np.zeros(n_elems)

    elems = abq_dict.keys()
    for i, elem in enumerate(elems):
        cogs_plot[i] = [
            abq_dict[elem]["centroid"][0],
            abq_dict[elem]["centroid"][1],
            abq_dict[elem]["centroid"][2],
        ]
        evect1[i] = abq_dict[elem]["mm"][0]
        evect2[i] = abq_dict[elem]["mm"][1]
        evect3[i] = abq_dict[elem]["mm"][2]
        eval1[i] = abq_dict[elem]["m"][0]
        eval2[i] = abq_dict[elem]["m"][1]
        eval3[i] = abq_dict[elem]["m"][2]

    savepath = (
        Path(cfg.paths.feadir)
        / cfg.simulations.folder_id[sample]
        / f"{sample}_{cfg.version.current_version}"
    )

    quiver_3d_MSL(eval1, evect1, cogs_plot, str(savepath.resolve()) + "_MSL_1.png")
    quiver_3d_MSL(eval2, evect2, cogs_plot, str(savepath.resolve()) + "_MSL_2.png")
    quiver_3d_MSL(eval3, evect3, cogs_plot, str(savepath.resolve()) + "_MSL_3.png")
    return None
