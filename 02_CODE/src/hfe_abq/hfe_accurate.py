"""
Script runs ACCURATE pipeline, converted from Denis's Bash script.

Author: Denis Schenk, ARTORG Center for Biomedical Engineering Research, SITEM Insel, University of Bern
Maintained by: Simone Poncioni, ARTORG Center for Biomedical Engineering Research, SITEM Insel, University of Bern
Date: April 2021
Latest update: 16.11.2023

UPDATES:
- Updated to run multiple simulations independently in parallel (POS)
"""

import os
from pathlib import Path
from time import time

import hfe_abq.aim2fe as aim2fe
import hfe_utils.imutils as imutils
import yaml
from hfe_utils.io_utils import print_mem_usage, write_timing_summary

os.environ["NUMEXPR_MAX_THREADS"] = "16"
"""
import shutil
import sys
from importlib import reload

import create_loadcases_SA
import io_utils_SA as io_utils
import numpy as np
import optimization as optimization
import postprocessing_SA as post
import print_optim_report as por
import simulation
import utils_SA as utils
"""

# flake8: noqa: E402, W503


def pipeline_hfe(cfg, folder_id, grayscale_filename):
    # timing
    time_record = {}
    start_full = time()
    start_sample = time()
    print_mem_usage()

    # Sets paths
    workdir = cfg.paths.workdir
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

    """
    if cfg.image_processing.BVTVd_comparison is True:
        bone = imutils.compute_bvtv_d_seg(bone, grayscale_filename)

    if cfg.meshing.meshing == "spline":
        inputfile = str(abq_inp_path.resolve())

    # creating 6 loadcases
    reload(create_loadcases_SA)
    create_loadcases_SA.update_bc_files(config)
    create_loadcases_SA.create_canonical_loadcases(config, inputfile)

    # 3.2) FZ_MAX
    optim = {}
    reload(create_loadcases_SA)

    # check for size of image to determine wether it's a radius or tibia, else use config["site_bone"]
    cogs_dict = bone["cogs"]

    DIMZ = 0.0
    for arr in cogs_dict.values():
        for arr2 in arr.values():
            if arr2[-1] > DIMZ:
                DIMZ = arr2[-1]

    if DIMZ <= 21.0:  # TODO: check this value
        # assume double stack radius
        create_loadcases_SA.create_loadcases_fm_max_no_psl_radius(
            config, sample, "FZ_MAX"
        )
    elif DIMZ <= 40.0:  # TODO: check this value
        # assume triple stack tibia
        create_loadcases_SA.create_loadcases_fm_max_no_psl_tibia(
            config, sample, "FZ_MAX"
        )
    else:
        print(f"Image size {DIMZ} not recognized, using config['site_bone']")
        if config["site_bone"].lower() == "radius":
            create_loadcases_SA.create_loadcases_fm_max_no_psl_radius(
                config, sample, "FZ_MAX"
            )
        elif config["site_bone"].lower() == "tibia":
            create_loadcases_SA.create_loadcases_fm_max_no_psl_tibia(
                config, sample, "FZ_MAX"
            )
        else:
            raise ValueError(
                "Site bone in config was not properly defined. Use keywords [Radius] or [Tibia]"
            )

    start_simulation = time.time()
    if do_simulation:
        try:
            simulation.simulate_loadcases_psl(config, sample, inputfile, umat, "FZ_MAX")
            end_simulation = time.time()
        except Exception:
            print("Simulation of FZ_MAX loadcase resulted in error")
            print(sys.stderr)
            end_simulation = time.time()
            pass
    else:
        end_simulation = time.time()
    time_record["simulation"] = end_simulation - start_simulation

    optim = post.datfilereader_psl(config, sample, optim, "FZ_MAX")

    # timing
    end_sample = time.time()
    optim["processing_time"] = end_sample - start_sample
    time_record[sample] = end_sample - start_sample

    reload(por)
    reload(utils)
    optim = por.compute_optim_report_variables_no_psl(config, sample, optim, bone)
    bone = por.compute_bone_report_variables_no_psl(bone)
    utils.write_data_summary_no_psl(
        config,
        optim,
        bone,
        sample,
        DOFs=bone["degrees_of_freedom"],
        time_sim=time_record[sample],
    )

    if config["delete_odb"]:
        odbfilename = "{}_FZ_MAX_{}.odb".format(sample, current_version[0:2])
        odbfile = os.path.join(feadir, folder, odbfilename)
        os.remove(odbfile)

    sampledir_parent = Path(sampledir).parent

    try:
        # move whole content of feadir to sampledir except subdirectories
        for file in os.listdir(sampledir_parent):
            if os.path.isfile(os.path.join(sampledir_parent, file)):
                shutil.move(os.path.join(sampledir_parent, file), sampledir)
    except Exception:
        current_time = time.strftime("%Y%m%d-%H%M%S")
        child_dir_time_path = (
            Path(sampledir) / f"simulation_current_time_{current_time}"
        )
        child_dir_time_path.mkdir(parents=True, exist_ok=True)
        print(f"File in this location already exists, moving to {child_dir_time_path}")
        for file in os.listdir(sampledir_parent):
            if os.path.isfile(os.path.join(sampledir_parent, file)):
                shutil.move(os.path.join(sampledir_parent, file), child_dir_time_path)

    """
    end_full = time()
    time_record["full"] = end_full - start_full
    summary_path = Path(
        sumdir / str(grayscale_filename + "_V_" + current_version + "_summary.txt")
    )

    print(yaml.dump(time_record, default_flow_style=False))

    write_timing_summary(cfg, grayscale_filename, time_record)
    return time_record, summary_path
