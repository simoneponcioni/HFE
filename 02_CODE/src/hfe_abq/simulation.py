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
    Run Abaqus simulation from os subprocess.

    Args:
        config: configuration dictionary
        sample (str): sample number
        inputfile (str): path of input file
        umat (str): path of UMAT subroutine
        loadcase (str): string defining the load case

    Returns:
        None
    """
    
    setup_env_cmd = 'module load intel-compilers/2021.2.0'
    
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
    # command = (
    #     f"{setup_env_cmd} && "
    #     f"{ABAQUS} job={job} inp={inputfile} cpus={NPROCS} memory={RAM} "
    #     f"user={umat} scratch={SCRATCH} ask_delete=OFF verbose=3 -interactive"
    # )
    command = (
        f"{ABAQUS} job={job} inp={inputfile} cpus={NPROCS} memory={RAM} "
        f"user={umat} scratch={SCRATCH} ask_delete=OFF verbose=3 -interactive"
    )
    logger.info(command)
    try:
        result = subprocess.run(command, capture_output=True, text=True, shell=True)
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
    
    odb_path = simdir / (job + ".odb")
    return odb_path
