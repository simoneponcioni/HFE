"""
Script runs ACCURATE pipeline, converted from Denis's Bash script.

Author: Denis Schenk, ARTORG Center for Biomedical Engineering Research, SITEM Insel, University of Bern
Maintained by: Simone Poncioni, ARTORG Center for Biomedical Engineering Research, SITEM Insel, University of Bern
Date: April 2021
Latest update: 16.11.2023

UPDATES:
- Updated to run multiple simulations independently in parallel (POS)
"""

import logging
import os
import sys
from pathlib import Path
from shutil import move
from time import time

import hfe_abq.aim2fe as aim2fe
import hfe_abq.create_loadcases as create_loadcases
import hfe_abq.simulation as simulation
import hfe_accurate.postprocessing as postprocessing
import hfe_utils.imutils as imutils
import hfe_utils.print_optim_report as por
import numpy as np
import yaml
from hfe_utils.io_utils import print_mem_usage, write_timing_summary

os.environ["NUMEXPR_MAX_THREADS"] = "16"

LOGGING_NAME = "HFE-ACCURATE"
logger = logging.getLogger(LOGGING_NAME)

# flake8: noqa: E402, W503


def pipeline_hfe(cfg, folder_id, grayscale_filename):
    """
    # TODO: reactivate for sensitivity analysis
    n_sim = int(10)  # has to match sweep in config
    # min = 3, 5, 1, 7
    # max = 20, 50, 10, 50 did not work, reducing to 20, 40, 10, 40
    n_elms_longitudinal = np.linspace(5, 15, n_sim, dtype=int)
    n_elms_transverse_trab = np.linspace(5, 30, n_sim, dtype=int)
    n_elms_transverse_cort = np.linspace(2, 8, n_sim, dtype=int)
    n_radial = np.linspace(10, 40, n_sim, dtype=int)

    # update meshing settings with sweep factor for sensitivity analysis
    sweep = cfg.meshing_settings.sweep_factor
    cfg.meshing_settings.n_elms_longitudinal = int(
        n_elms_longitudinal[sweep - 1].item()
    )
    cfg.meshing_settings.n_elms_transverse_trab = int(
        n_elms_transverse_trab[sweep - 1].item()
    )
    cfg.meshing_settings.n_elms_transverse_cort = int(
        n_elms_transverse_cort[sweep - 1].item()
    )
    cfg.meshing_settings.n_elms_radial = int(n_radial[sweep - 1].item())
    """

    # timing
    time_record = {}
    start_full = time()
    start_sample = time()
    print_mem_usage()

    # Sets paths
    workdir = cfg.socket_paths.workdir
    origaimdir = Path(workdir, cfg.paths.origaimdir)
    aimdir = Path(workdir, cfg.paths.aimdir)
    feadir = Path(workdir, cfg.paths.feadir)
    umat = Path(workdir, cfg.abaqus.umat)
    sumdir = Path(workdir, cfg.paths.sumdir)
    sumdir.mkdir(parents=True, exist_ok=True)
    feadir.mkdir(parents=True, exist_ok=True)

    # sample = grayscale_filename  # TODO: refactoring this
    current_version = cfg.version.current_version

    sampledir = Path(feadir) / cfg.simulations.folder_id[grayscale_filename]
    inputfilename = f"{grayscale_filename}.inp".format(
        grayscale_filename, current_version
    )
    inputfile = sampledir / inputfilename
    sampledir.mkdir(parents=True, exist_ok=True)

    bone = {}
    bone, abq_inp_path = aim2fe.aim2fe_psl(cfg, grayscale_filename)

    if cfg.image_processing.BVTVd_comparison is True:
        bone = imutils.compute_bvtv_d_seg(bone, grayscale_filename)

    if cfg.mesher.meshing == "spline":
        inputfile = str(abq_inp_path.resolve())

    # 3.2) FZ_MAX
    cogs_dict = bone["cogs"]

    DIMZ = 0.0
    for arr in cogs_dict.values():
        for arr2 in arr.values():
            if arr2[-1] > DIMZ:
                DIMZ = arr2[-1]

    # create_loadcases.create_loadcase_fz_max(cfg, grayscale_filename, "FZ_MAX")

    start_simulation = time()
    try:
        simulation.simulate_loadcase(cfg, grayscale_filename, inputfile, umat, "")
        end_simulation = time()
    except Exception:
        logger.error("Simulation of FZ_MAX loadcase resulted in error")
        logger.error(sys.stderr)
        end_simulation = time()
        pass
    else:
        end_simulation = time()
    time_record["simulation"] = end_simulation - start_simulation

    optim = {}
    optim = postprocessing.datfilereader_psl(cfg, grayscale_filename, optim, "FZ_MAX")

    # timing
    end_sample = time()
    optim["processing_time"] = end_sample - start_sample
    time_record[grayscale_filename] = end_sample - start_sample

    optim = por.compute_optim_report_variables(optim)
    bone = por.compute_bone_report_variables_no_psl(bone)

    # only for sensitivity analysis
    mesh_parameters_dict = {
        "n_elms_longitudinal": cfg.meshing_settings.n_elms_longitudinal,
        "n_elms_transverse_trab": cfg.meshing_settings.n_elms_transverse_trab,
        "n_elms_transverse_cort": cfg.meshing_settings.n_elms_transverse_cort,
        "n_elms_radial": cfg.meshing_settings.n_elms_radial,
    }
    postprocessing.write_data_summary(
        cfg,
        optim,
        bone,
        grayscale_filename,
        mesh_parameters_dict,
        DOFs=bone["degrees_of_freedom"],
        time_sim=time_record[grayscale_filename],
    )

    if cfg.abaqus.delete_odb:
        odbfilename = "{}_FZ_MAX_{}.odb".format(
            grayscale_filename, current_version[0:2]
        )
        odbfile = os.path.join(feadir, folder_id, odbfilename)
        os.remove(odbfile)

    sampledir_parent = Path(sampledir).parent

    try:
        # move whole content of feadir to sampledir except subdirectories
        for file in os.listdir(sampledir_parent):
            if os.path.isfile(os.path.join(sampledir_parent, file)):
                move(os.path.join(sampledir_parent, file), sampledir)
    except Exception:
        current_time = time.strftime("%Y%m%d-%H%M%S")
        child_dir_time_path = (
            Path(sampledir) / f"simulation_current_time_{current_time}"
        )
        child_dir_time_path.mkdir(parents=True, exist_ok=True)
        logger.info(
            f"File in this location already exists, moving to {child_dir_time_path}"
        )
        for file in os.listdir(sampledir_parent):
            if os.path.isfile(os.path.join(sampledir_parent, file)):
                move(os.path.join(sampledir_parent, file), child_dir_time_path)

    end_full = time()
    time_record["full"] = end_full - start_full
    summary_path = Path(
        sumdir / str(grayscale_filename + "_V_" + current_version + "_summary.txt")
    )

    print(yaml.dump(time_record, default_flow_style=False))

    write_timing_summary(cfg, grayscale_filename, time_record)
    return time_record, summary_path
