import logging
import os
import subprocess
import sys
import traceback
from pathlib import Path
import time

LOGGING_NAME = "HFE-ACCURATE"
logger = logging.getLogger(LOGGING_NAME)
logger.propagate = False


def simulate_loadcase(cfg, sample: str, inputfile: str, umat: str, loadcase: str):
    """
    Run abaqus simulation from os subprocess.

    Args:
        config: configuration dictionary
        sample (str): sample number
        inputfile (str): path of input file
        umat (str): path of UMAT subroutine
        loadcase (str): string defining the load case

    Returns:
        None
    """
    ABAQUS = cfg.solver.abaqus
    NPROCS = cfg.abaqus.abaqus_nprocs
    RAM = cfg.abaqus.abaqus_memory
    SCRATCH = cfg.socket_paths.scratchdir
    # if loadcase is not an empty string, then:
    if not loadcase:
        job = sample + "_" + cfg.version.current_version[0:2]
    else:
        job = sample + "_" + loadcase + "_" + cfg.version.current_version[0:2]

    feadir = Path(cfg.paths.feadir)
    folder_id = Path(cfg.simulations.folder_id[sample])
    simdir = Path(feadir / folder_id)
    simdir.mkdir(exist_ok=True)
    basepath = os.getcwd()

    os.chdir(simdir)
    command = [
        ABAQUS, 
        f"job={job}", 
        f"inp={inputfile}", 
        f"cpus={NPROCS}", 
        f"memory={RAM}", 
        f"user={umat}", 
        f"scratch={SCRATCH}", 
        "ask_delete=OFF", 
        "verbose=3", 
        "-interactive"
    ]
    logger.info(" ".join(command))
    try:
        result = subprocess.run(command, capture_output=True, text=True)
        logger.info(result.stdout)
        if result.returncode == 0:
            logger.info("Abaqus simulation completed successfully")
        else:
            logger.error("Abaqus simulation failed with return code %d", result.returncode)
            logger.error(result.stderr)
    except Exception as e:
        logger.error("Simulation of FZ_MAX loadcase resulted in error")
        logger.error(e)
        logger.error(traceback.format_exc())
    finally:
        os.chdir(basepath)
        time.sleep(600) # TODO: remove this as soon as ubelix issue is solved (POS, 23.08.2024)
    
    odb_path = simdir / (job + ".odb")
    return odb_path