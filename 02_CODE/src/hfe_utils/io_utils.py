import os
from pathlib import Path

import psutil
from omegaconf import OmegaConf

# flake8: noqa: E501


def print_mem_usage():
    process = psutil.Process(os.getpid())
    mem_info = process.memory_info()
    mem_used_gb = mem_info.rss / (1024**3)  # Convert bytes to GB
    print(f"Memory usage:\t\t{mem_used_gb:.2f} (GB)")


def log_append_processingtime(filename, time):
    SUMname = filename
    time_summary = "\n".join(
        [
            "Summary Processing Time",
            "Full processing time           : {:.3f} [s]".format(time),
            "****************************************************************",
        ]
    )
    print(time_summary)

    with open(SUMname, "a") as sumUpdate:
        sumUpdate.write("\n")
        sumUpdate.write(time_summary)
    print("... added processing time to summary file")


def write_timing_summary(cfg, sample: str, time: dict):
    """
    Writes a summary of processing times for all samples in config
    to summaries folder.
    Parameters
    ----------
    path    path to store (summaries)
    time    dict with full processing time and simulation processing time

    Returns
    -------
    writes txt file
    """

    # file_path with pathlib from confing
    file_path = (
        Path(cfg.paths.sumdir)
        / f"{cfg.version.current_version}_processing_time_summary.csv"
    )

    with open(file_path, "w") as f:
        print(time, file=f)


def set_filenames(cfg, sample, pipeline="fast", origaim_separate=True):
    """
    Set filenames for each grayscale file.
    Filenames depend on pipeline (fast/accurate).
    Always:
        - native image for header
        - BMD or Native image croppped to ROI
    additional for fast pipeline:
        - periosteal mask
    additional for accurate pipeline:
        - trabecular mask
        - cortical mask
        - two-phase segmentation
    """

    # Always, not depending on nphase
    current_version = cfg.version.current_version
    folder_id = cfg.simulations.folder_id
    folder = folder_id[sample]
    feadir = Path(cfg.paths.feadir) / folder
    aimdir = Path(cfg.paths.aimdir) / folder
    origaimdir = Path(cfg.paths.origaimdir) / folder

    filename_postfix_bmd = cfg.filenames.filename_postfix_bmd

    filename = {}
    filename["FILEBMD"] = f"{sample}{filename_postfix_bmd}"
    if origaim_separate:
        filename["FILEGRAY"] = Path(sample).with_suffix(".AIM")
    else:
        filename["FILEGRAY"] = filename["FILEBMD"]
    filename["RAWname"] = str(Path(origaimdir) / filename["FILEGRAY"])
    filename["BMDname"] = str(Path(origaimdir) / filename["FILEBMD"])
    filename["boundary_conditions"] = cfg.paths.boundary_conditions

    # additional for fast pipeline
    if pipeline == "fast":
        filename_postfix_mask = cfg.filenames.filename_postfix_mask
        filename["FILEMASK"] = f"{sample}{filename_postfix_mask}"
        filename["MASKname"] = str(Path(origaimdir) / filename["FILEMASK"])

    # Additionl for accurate pipeline
    if pipeline == "accurate":
        filename_postfix_trab_mask = cfg.filenames.filename_postfix_trab_mask
        filename_postfix_cort_mask = cfg.filenames.filename_postfix_cort_mask
        filename_postfix_seg = cfg.filenames.filename_postfix_seg

        filename["FILEMASKCORT"] = f"{sample}{filename_postfix_cort_mask}"
        filename["FILEMASKTRAB"] = f"{sample}{filename_postfix_trab_mask}"
        filename["FILESEG"] = f"{sample}{filename_postfix_seg}"

        filename["CORTMASKname"] = str(Path(origaimdir) / filename["FILEMASKCORT"])
        filename["TRABMASKname"] = str(Path(origaimdir) / filename["FILEMASKTRAB"])
        filename["SEGname"] = str(Path(origaimdir) / filename["FILESEG"])

        if cfg.image_processing.mask_separate is False:
            filename_postfix_mask = cfg.filenames.filename_postfix_mask
            filename["FILEMASK"] = f"{sample}{filename_postfix_mask}"
            filename["MASKname"] = Path(origaimdir) / filename["FILEMASK"]

    # General filenames
    print(filename["BMDname"])
    new_filename = f"{sample}_V_{current_version}.inp"
    filename["INPname"] = str(Path(feadir) / new_filename)
    filename["VTKname"] = str(Path(aimdir) / folder / new_filename)

    new_filename = f"{sample}_V_{current_version}_summary.txt"
    filename["SUMname"] = str(Path(feadir) / new_filename)

    new_filename = f"{sample}_V_{current_version}_BPVb"
    filename["VER_BPVname"] = str(Path(feadir) / new_filename)

    return filename


def hydra_update_cfg_key(cfg, key, value):
    # Make the config mutable
    OmegaConf.set_struct(cfg, False)

    # Split the key and access each level of the configuration separately
    keys = key.split(".")
    for k in keys[:-1]:
        cfg = cfg[k]

    setattr(cfg, keys[-1], value)

    # Make the config read-only again
    OmegaConf.set_struct(cfg, True)
    return None
