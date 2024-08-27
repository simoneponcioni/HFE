# Recovery of Image properties within spline hex mesh
# @Author:  Simone Poncioni, MSB
# @Date:    18.06.2024

import numpy as np
from pathlib import Path
import SimpleITK as sitk
import gmsh


# * Cortical volume of mask vs cortical volume of mesh
# ---
def calculate_mask_volume(mhd_path: Path):
    def volume(mask_image):
        space = mask_image.GetSpacing()  # image spacing
        voxel = np.prod(space)  # voxel volume
        img = sitk.GetArrayFromImage(mask_image)
        vol = voxel * np.count_nonzero(img)
        return vol

    imsitk = sitk.ReadImage(str(mhd_path))
    vol = volume(imsitk)
    return vol


def calculate_mesh_volume(mesh_path: Path):
    gmsh.initialize()
    gmsh.open(str(mesh_path))

    gmsh.plugin.run("NewView")
    gmsh.plugin.setNumber("MeshVolume", "Dimension", 3)
    gmsh.plugin.setNumber(
        "MeshVolume", "PhysicalGroup", 1
    )  # Physical group 1 is the cortical compartment
    gmsh.plugin.run("MeshVolume")

    views = gmsh.view.getTags()
    _, _, data = gmsh.view.getListData(views[-1])
    volume = data[0][-1]

    gmsh.clear()
    gmsh.finalize()
    return volume


def difference_volume_percent(mhd_path: Path, mesh_path: Path):
    mask_vol = calculate_mask_volume(mhd_path)
    mesh_vol = calculate_mesh_volume(mesh_path)
    diff = (mesh_vol - mask_vol) / mask_vol * 100
    return diff


def calc_diff_auto():
    vol_path = Path("01_DATA/438_L_71_F/C0003104_CORTMASK.mhd")
    mesh_path = Path("03_MESH/C0003104sweep_1/C0003104sweep_1.msh")
    diff_test = difference_volume_percent(vol_path, mesh_path)
    print(f"\n\nDifference in volume: {diff_test:.3f}%\n\n")


# ---


def main():
    # get all meshes in subdir:
    parent_dir = Path("03_MESH")
    # for each subdir, get the mesh with suffix '.msh'
    res_dict = {}
    for subdir in parent_dir.iterdir():
        if subdir.is_dir():
            for mesh in subdir.iterdir():
                if mesh.suffix == ".msh":
                    print(f"Processing {mesh}")
                    vol = calculate_mesh_volume(mesh)
                    res_dict[str(mesh.name).split("s")[0]] = vol
    print(res_dict)
    # save dict as csv
    with open("mesh_volumes.csv", "w") as f:
        for key in res_dict.keys():
            f.write("%s,%s\n" % (key, res_dict[key]))


if __name__ == "__main__":
    main()
