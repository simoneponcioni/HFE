import numpy as np
import pyvista as pv
from pathlib import Path


def list_vtus(basepath: Path):
    vtu_list = basepath.rglob("*_with_data.vtu")
    return list(vtu_list)


def read_rho_vtu(vtu_path: Path):
    vtu = pv.read(str(vtu_path))
    bvtvt = vtu.cell_data["SDV_BVTVT_Centroid"]
    avg_bvtvt = np.mean(bvtvt)
    bvtvc = vtu.cell_data["SDV_BVTVC_Centroid"]
    avg_bvtvc = np.mean(bvtvc)
    return avg_bvtvt, avg_bvtvc


def append_csv(csv_path: Path, vtu_path: Path, bvtvt: float, bvtvc: float):
    vtu_name = vtu_path.stem.split('_')[0]
    with open(csv_path, "a") as f:
        f.write(f"{vtu_name},{bvtvt},{bvtvc}\n")


def main():
    basepath = Path("/storage/workspaces/artorg_msb/hpc_abaqus/poncioni/HFE/04_SIMULATIONS/REPRO_PAPER/IMAGES")
    csv_path = Path.cwd() / "rho.csv"

    vtu_l = list_vtus(basepath)

    for vtu_path in vtu_l:
        bvtvt, bvtvc = read_rho_vtu(vtu_path)
        append_csv(csv_path, vtu_path, bvtvt, bvtvc)

    return None


if __name__ == "__main__":
    main()
