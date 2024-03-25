import os
import sys
import traceback
from pathlib import Path


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
    SCRATCH = cfg.abaqus.scratchdir
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
    command = str(
        "%s job=%s inp=%s cpus=%d memory=%d user=%s scratch=%s ask_delete=OFF -interactive"
        % (ABAQUS, job, inputfile, NPROCS, RAM, umat, SCRATCH)
    )
    print(command)
    try:
        os.system(command)
    except Exception as e:
        print("Simulation of FZ_MAX loadcase resulted in error")
        print(e)
        print(traceback.format_exc())
        print(sys.stderr)
        pass
    os.chdir(basepath)
    return None
