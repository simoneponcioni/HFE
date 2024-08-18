from pathlib import Path
import pyvista as pv


def get_vtu_paths(base_path: Path):
    return [mesh for mesh in base_path.rglob("*.vtu")]


def extract_prop(mesh_paths: list, prop_name: str):
    # create a loop and extract the cell data for each mesh
    props = []
    for mesh_path in mesh_paths:
        print(f"{mesh_path}")
        mesh = pv.read(mesh_path)
        props.append(mesh.cell_data[prop_name])
    # now we have a list of props, we can create a new mesh with the average property
    average_prop = sum(props) / len(props)
    return average_prop


def save_with_prop(ssm_path: Path, prop: pv.PolyData, prop_name: str):
    mesh = pv.read(ssm_path)
    mesh.cell_data[prop_name] = prop
    mesh_path = ssm_path.parent / f"{ssm_path.stem}_{prop_name}.vtu"
    mesh.save(mesh_path)


def main():
    # base_path = Path(
    #     "/storage/workspaces/artorg_msb/hpc_abaqus/poncioni/HFE/04_SIMULATIONS/TIBIA/SENER"
    # )
    base_path = Path("/home/simoneponcioni/Desktop/SENER/vtus")
    ssm_path = base_path.parent / "tibia-ssm.vtk"
    prop_name = "SENER_Centroid"
    vtu_paths = get_vtu_paths(base_path)
    print(vtu_paths)
    average_prop = extract_prop(vtu_paths, prop_name)
    save_with_prop(ssm_path, average_prop, prop_name)
    return None


if __name__ == "__main__":
    main()
