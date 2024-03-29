import logging
from typing import Literal

import fast_simplification as fs  # type: ignore
import pyvista as pv
import vtk  # type: ignore
from hfe_utils.io_utils import timeit

LOGGING_NAME = "HFE-ACCURATE"
logger = logging.getLogger(LOGGING_NAME)
logger.setLevel(logging.INFO)
handler = logging.FileHandler("./pipeline_runner.log")
handler.setLevel(logging.INFO)

console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
handler.setFormatter(formatter)
console_handler.setFormatter(formatter)

logger.addHandler(handler)
logger.addHandler(console_handler)


@timeit
def surface_nets(
    imvtk,
    output_mesh_type: Literal["quads", "tri"] = "tri",
    output_style: Literal["default", "boundary"] = "default",
    smoothing: bool = False,
    decimate: bool = False,
    smoothing_num_iterations: int = 50,
    target_reduction: float = 0.9,
):
    """
    Applies the Surface Nets algorithm to a given vtkImageData object.

    Parameters:
    imvtk (vtk.vtkImageData): The input image data.
    output_mesh_type (str, optional): The type of mesh to output.
        Can be either "quads" or "tri". Defaults to "tri".
    output_style (str, optional): The style of the output.
        Can be either "default" or "boundary". Defaults to "default".
    smoothing (bool, optional): Whether to apply smoothing to the output.
        Defaults to False.
    smoothing_num_iterations (int, optional): The number of iterations
        to perform if smoothing is enabled. Defaults to 50.
    smoothing_relaxation_factor (float, optional):
        The relaxation factor to use if smoothing is enabled. Defaults to 0.5.
    smoothing_constraint_distance (float, optional):
        The constraint distance to use if smoothing is enabled. Defaults to 1.

    Raises:
    ValueError: If an invalid output mesh type or output style is provided.
    NotImplementedError: If the selected output style is not implemented.

    Returns:
    pv.core.pointset.UnstructuredGrid: Mesh after applying Surface Nets.
    """
    TARGET_REDUCTION = target_reduction

    num_labels = int(imvtk.GetPointData().GetScalars().GetRange()[1])

    surfnets = vtk.vtkSurfaceNets3D()
    surfnets.SetInputData(imvtk)

    if num_labels is not None:
        surfnets.GenerateLabels(num_labels, 1, num_labels)

    if output_mesh_type == "quads":
        surfnets.SetOutputMeshTypeToQuads()
    elif output_mesh_type == "tri":
        surfnets.SetOutputMeshTypeToTriangles()
    else:
        raise ValueError(
            f'Invalid output mesh type "{output_mesh_type}", use "quads" or "tri"'
        )
    if output_style == "default":
        surfnets.SetOutputStyleToDefault()
    elif output_style == "boundary":
        surfnets.SetOutputStyleToBoundary()
    elif output_style == "selected":
        raise NotImplementedError(f'Output style "{output_style}" is not implemented')
    else:
        raise ValueError(
            f'Invalid output style "{output_style}", use "default" or "boundary"'
        )

    surfnets.Update()
    mesh = pv.wrap(surfnets.GetOutput())

    # Decimation
    if decimate:
        mesh = fs.simplify_mesh(
            mesh, target_reduction=TARGET_REDUCTION, agg=8, verbose=True
        )

    # Smoothing
    if smoothing:
        mesh = mesh.smooth_taubin(n_iter=smoothing_num_iterations, pass_band=0.5)

    # save mesh for debugging
    # timenow = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    # mesh.save(f"mesh_{timenow}.vtk")

    logger.info("1/6 STL file creation finished")
    return mesh
