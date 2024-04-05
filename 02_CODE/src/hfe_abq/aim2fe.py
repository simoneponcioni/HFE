"""
Converts an AIM file to a HDF5 file.
Adapted from the script from Denis Schenk, ISTB, University of Bern.

Author: Jarunan Panyasantisuk, ETH Scientific IT Services.
Date: 13 November 2018.

Maintained by: Simone Poncioni, ARTORG Center for Biomedical Engineering Research, SITEM Insel, University of Bern
Date: March 2024
"""

import gc
import logging
import pickle
import threading
from pathlib import Path
from time import sleep

import hfe_accurate.material_mapping as material_mapping
import hfe_accurate.preprocessing as preprocessing
import hfe_utils.imutils as imutils
import hfe_utils.io_utils as io_utils
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import SimpleITK as sitk
import yaml
from mpl_toolkits.axes_grid1 import make_axes_locatable  # type: ignore
from pyhexspline.spline_mesher import HexMesh  # type: ignore

LOGGING_NAME = "HFE-ACCURATE"
logger = logging.getLogger(LOGGING_NAME)


matplotlib.use("TkAgg")


def _helper_store_bone_dict(bone: dict, basepath: Path, _mesh: str):
    """
    Helper function to store the bone dict as a pickle file
    Can be removed after testing (POS, 02.08.2023)
    """
    BVTVscaled = bone["BVTVscaled"]
    with open(basepath / f"{_mesh}_BVTVscaled.pkl", "wb") as a:
        pickle.dump(BVTVscaled, a)

    BMDscaled = bone["BMDscaled"]
    with open(basepath / f"{_mesh}_BMDscaled.pkl", "wb") as b:
        pickle.dump(BMDscaled, b)

    CORTMASK_array = bone["CORTMASK_array"]
    with open(basepath / f"{_mesh}_CORTMASK_array.pkl", "wb") as c:
        pickle.dump(CORTMASK_array, c)

    TRABMASK_array = bone["TRABMASK_array"]
    with open(basepath / f"{_mesh}_TRABMASK_array.pkl", "wb") as d:
        pickle.dump(TRABMASK_array, d)

    SEG_array = bone["SEG_array"]
    with open(basepath / f"{_mesh}_SEG_array.pkl", "wb") as e:
        pickle.dump(SEG_array, e)

    FEelSize = bone["FEelSize"]
    with open(basepath / f"{_mesh}_FEelSize.pkl", "wb") as f:
        pickle.dump(FEelSize, f)

    Spacing = bone["Spacing"]
    with open(basepath / f"{_mesh}_Spacing.pkl", "wb") as g:
        pickle.dump(Spacing, g)

    elems = bone["elms"]
    with open(basepath / f"{_mesh}_elems.pkl", "wb") as h:
        pickle.dump(elems, h)

    nodes = bone["nodes"]
    with open(basepath / f"{_mesh}_nodes.pkl", "wb") as i:
        pickle.dump(nodes, i)

    MSL_kernel_list_cort = bone["MSL_kernel_list_cort"]
    with open(basepath / f"{_mesh}_MSL_kernel_list_cort.pkl", "wb") as k:
        pickle.dump(MSL_kernel_list_cort, k)

    MSL_kernel_list_trab = bone["MSL_kernel_list_trab"]
    with open(basepath / f"{_mesh}_MSL_kernel_list_trab.pkl", "wb") as file_handler:
        pickle.dump(MSL_kernel_list_trab, file_handler)

    MESH = bone["MESH"]
    with open(basepath / f"{_mesh}_MESH.pkl", "wb") as m:
        pickle.dump(MESH, m)
    return None


def save_image_with_colorbar(data, output_path):
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
    plt.savefig(output_path, dpi=300)
    plt.clf()


def aim2fe_psl(cfg, sample):
    """
    Wrapper that converts AIM image to Abaqus INP.

    Args:
        config (dict): dictionary of configuration parameters
        sample (_type_): _description_

    Raises:
        TypeError: _description_

    Returns:
        _type_: _description_
    """

    # For Hosseini Dataset, Image parameters are read from original aim, not from processed BMD file, as there
    # they were deleted by medtool pre-processing
    origaim_separate_bool = cfg.image_processing.origaim_separate

    filenames = io_utils.set_filenames(
        cfg, sample, pipeline="accurate", origaim_separate=origaim_separate_bool
    )

    print(yaml.dump(filenames, default_flow_style=False))
    io_utils.print_mem_usage()

    # 2 Read AIM images and image parameters
    # ---------------------------------------------------------------------------------
    bone = {}
    bone["sample"] = str(sample)

    spacing, scaling, slope, intercept = imutils.read_img_param(filenames)
    bone["Spacing"] = spacing
    bone["Scaling"] = scaling
    bone["Slope"] = slope
    bone["Intercept"] = intercept

    if cfg.image_processing.mask_separate is True:
        image_list = ["BMD", "SEG", "CORTMASK", "TRABMASK"]
        threads = []
        lock = threading.Lock()
        for item in image_list:
            t = threading.Thread(
                target=imutils.read_aim, args=(item, filenames, bone, lock)
            )
            threads.append(t)
            sleep(0.1)  # to avoid overloading the print statements
            t.start()
        for t in threads:
            t.join()
    else:
        image_list = ["BMD", "SEG"]
        for _, item in enumerate(image_list):
            bone = io_utils.read_aim(item, filenames, bone)
        bone = imutils.read_aim_mask_combined("MASK", filenames, bone)

    image_list = ["BMD", "SEG", "CORTMASK", "TRABMASK"]
    for _, item in enumerate(image_list):
        bone = imutils.adjust_image_size(item, bone, cfg, imutils.CropType.crop)

    # Save images with colorbar
    imutils.save_images_with_colorbar(cfg, sample, bone)

    # 3 Prepare material mapping
    # ---------------------------------------------------------------------------------

    IMTYPE = cfg.image_processing.imtype
    io_utils.print_mem_usage()
    BVTVscaled, BMDscaled, BVTVraw = preprocessing.calculate_bvtv(
        bone["Scaling"],
        bone["Slope"],
        bone["Intercept"],
        bone["BMD_array"],
        bone["CORTMASK_array"],
        bone["TRABMASK_array"],
        cfg,
        IMTYPE,
    )

    bone["BVTVscaled"] = BVTVscaled
    bone["BMDscaled"] = BMDscaled
    bone["BVTVraw"] = BVTVraw

    del BVTVscaled, BMDscaled, BVTVraw
    gc.collect()

    if cfg.mesher.meshing == "spline":
        cort_mask_np = bone["CORTMASK_array"]
        trab_mask_np = bone["TRABMASK_array"]
        masks = [cort_mask_np, trab_mask_np]
        mask_names = ["CORTMASK", "TRABMASK"]
        for mask, mask_name in zip(masks, mask_names):
            sitk_image = sitk.GetImageFromArray(mask)
            sitk_image = sitk.PermuteAxes(sitk_image, [2, 1, 0])
            sitk_image = sitk.Flip(sitk_image, [False, True, False])
            sitk_image.SetSpacing(bone["Spacing"])
            cortmask_path = (
                Path(cfg.paths.aimdir)
                / cfg.simulations.folder_id[sample]
                / f"{sample}_{mask_name}.mhd"
            )
            sitk.WriteImage(sitk_image, cortmask_path)

        # append sample filename to config_mesh["img_settings"]
        sample_n = str(sample) + f"sweep_{cfg.meshing_settings.sweep_factor}"
        io_utils.hydra_update_cfg_key(cfg, "img_settings.img_basefilename", sample_n)
        cfg.img_settings.img_basefilename = sample_n
        mesh = HexMesh(
            cfg.meshing_settings,
            cfg.img_settings,
            sitk_image=sitk_image,
        )

        (
            nodes,
            elms,
            nb_nodes,
            centroids_cort,
            centroids_trab,
            elm_vol_cort,
            elm_vol_trab,
            radius_roi_cort,
            radius_roi_trab,
            bnds_bot,
            bnds_top,
            reference_point_coord,
        ) = mesh.mesher()

        bone["nodes"] = nodes
        bone["elms"] = elms
        bone["degrees_of_freedom"] = len(nodes) * 6
        bone["elms_centroids_cort"] = centroids_cort
        bone["elms_centroids_trab"] = centroids_trab
        bone["elms_vol_cort"] = elm_vol_cort
        bone["elms_vol_trab"] = elm_vol_trab
        bone["bnds_bot"] = bnds_bot
        bone["bnds_top"] = bnds_top
        bone["reference_point_coord"] = reference_point_coord

        bone["elsets"] = []
        CoarseFactor = bone["FEelSize"][0] / bone["Spacing"][0]
        BVTVscaled_shape = bone["BVTVscaled"].shape
        bone["MESH"] = np.ones(
            ([int(dim) for dim in np.floor(np.array(BVTVscaled_shape) / CoarseFactor)])
        )

    # 4 Material mapping
    # ---------------------------------------------------------------------------------
    # Compute MSL kernel list
    if cfg.homogenization.fabric_type == "local":
        logger.info("Computing local MSL kernel list")
        bone = preprocessing.compute_msl_spline(bone, cfg)
    elif cfg.homogenization.fabric_type == "global":
        logger.info("Computing global MSL kernel list")
        pass
    else:
        raise ValueError("Fabric type not recognised")

    if cfg.mesher.meshing == "spline":
        # TODO: reactivate if you want pickled files (POS, 28.02.2024)
        # mesh_type = cfg.mesher.meshing
        # inp_filename = filenames["INPname"]
        # basepath = Path(inp_filename).parent
        # _helper_store_bone_dict(bone, basepath, _mesh=mesh_type)
        (
            bone,
            abq_dictionary,
            abq_inp_path,
        ) = material_mapping.material_mapping_spline(
            bone,
            cfg,
            filenames,
        )
        bone["abq_inp_path"] = abq_inp_path
    else:
        raise ValueError("Meshing type not recognised (ghost layer mode 1)")

    # 5 Compute and store summary and performance variables
    # ---------------------------------------------------------------------------------
    logger.info("Computing summary variables")
    summary_variables = preprocessing.set_summary_variables(bone)
    # io_utils.log_summary(bone, cfg, filenames, summary_variables)
    bone = dict(list(bone.items()) + list(summary_variables.items()))
    logger.info("Summary variables computed")

    if cfg.mesher.meshing == "spline":
        imutils.plot_MSL_fabric_spline(cfg, abq_dictionary, sample)
    else:
        raise NotImplementedError("Meshing type not recognised")
    return bone, abq_inp_path
