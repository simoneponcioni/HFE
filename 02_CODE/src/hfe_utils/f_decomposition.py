from pathlib import Path
import vtk
import numpy as np
import logging

LOGGING_NAME = "HFE-ACCURATE"
logger = logging.getLogger(LOGGING_NAME)
logger.propagate = False


def read_vtu(filepath):
    """
    Reads a .vtu file and returns the output of the vtkXMLUnstructuredGridReader.

    Args:
        filepath (str or pathlib.Path): The path to the .vtu file to read.

    Returns:
        vtk.vtkUnstructuredGrid: The output of the vtkXMLUnstructuredGridReader after reading the .vtu file.
    """
    file = vtk.vtkXMLUnstructuredGridReader()
    file.SetFileName(str(filepath))
    file.Update()
    return file.GetOutput()


def decomposition(F):
    """
    Performs a decomposition of the input matrix F into spherical compression and isovolumic deformation.

    Args:
        F (numpy.ndarray): A 3D numpy array with shape (3, 3, n), where n is the number of matrices to decompose.

    Returns:
        tuple: A tuple containing two 1D numpy arrays of length n. The first array represents the spherical
        compression of each matrix in F, and the second array represents the isovolumic deformation of each matrix.
    """
    spherical_compression = np.zeros(F.shape[-1])
    isovolumic_deformation = np.zeros(F.shape[-1])
    for i in range(F.shape[-1]):
        spherical_compression[i] = np.linalg.det(F[:, :, i])
        if spherical_compression[i] > 0:
            F_tilde = spherical_compression[i] ** (-1 / 3) * F[:, :, i]
            isovolumic_deformation[i] = np.linalg.norm(F_tilde)
        else:
            isovolumic_deformation[i] = 0.0
    return spherical_compression, isovolumic_deformation


def get_centroid_arrays(output_compression):
    """
    Extracts centroid arrays from the cell data of the given output compression.

    Args:
        output_compression (vtk.vtkMultiBlockDataSet): A VTK multi-block data set. The cell data of this
        data set is expected to contain arrays with names that include the word "Centroid".

    Returns:
        dict: A dictionary where keys are the names of the centroid arrays and values
        are the corresponding vtkDataArray objects. Only arrays with names that include the word "Centroid"
        are included in this dictionary.
    """
    arrays = output_compression.GetCellData()
    centroid_arrays = {
        arrays.GetArrayName(i): arrays.GetArray(i)
        for i in range(arrays.GetNumberOfArrays())
        if "Centroid" in arrays.GetArrayName(i)
    }
    return centroid_arrays


def get_F_matrix(centroid_arrays):
    """
    This function takes a dictionary of centroid arrays and constructs a 3x3 matrix F.

    Args:
        centroid_arrays (dict): A dictionary where keys are strings of the form "SDV_Fij_Centroid"
        and values are numpy arrays representing centroids. 'i' and 'j' in the key represent the
        row and column indices (ranging from 1 to 3) of the F matrix respectively.

    Returns:
        F (numpy.ndarray): A 3x3 matrix constructed from the centroid arrays. The shape of F is (3, 3, -1),
        where -1 implies that the size of the last dimension is inferred so that the total size remains constant.
    """
    F11 = centroid_arrays["SDV_F11_Centroid"]
    F12 = centroid_arrays["SDV_F12_Centroid"]
    F13 = centroid_arrays["SDV_F13_Centroid"]
    F21 = centroid_arrays["SDV_F21_Centroid"]
    F22 = centroid_arrays["SDV_F22_Centroid"]
    F23 = centroid_arrays["SDV_F23_Centroid"]
    F31 = centroid_arrays["SDV_F31_Centroid"]
    F32 = centroid_arrays["SDV_F32_Centroid"]
    F33 = centroid_arrays["SDV_F33_Centroid"]
    F = np.array([F11, F12, F13, F21, F22, F23, F31, F32, F33]).reshape(3, 3, -1)
    return F


def add_data_to_vtu(
    output_compression,
    spherical_compression,
    isovolumic_deformation,
    output_filepath,
):
    """
    Adds spherical compression and isovolumic deformation data to a vtkUnstructuredGrid object,
    and writes it to a .vtu file.

    Args:
        output_compression (vtk.vtkUnstructuredGrid): The vtkUnstructuredGrid object to which the data will be added.
        spherical_compression (list or numpy.ndarray): The spherical compression data to add.
        isovolumic_deformation (list or numpy.ndarray): The isovolumic deformation data to add.
        output_filepath (str or pathlib.Path): The path to the .vtu file to write.
    """
    # Create new arrays for spherical compression and isovolumic deformation
    spherical_compression_array = vtk.vtkFloatArray()
    spherical_compression_array.SetName("SphericalCompression")
    isovolumic_deformation_array = vtk.vtkFloatArray()
    isovolumic_deformation_array.SetName("IsovolumicDeformation")

    # Set the values of the arrays
    for i in range(len(spherical_compression)):
        spherical_compression_array.InsertNextValue(spherical_compression[i])
        isovolumic_deformation_array.InsertNextValue(isovolumic_deformation[i])

    output_compression.GetCellData().AddArray(spherical_compression_array)
    output_compression.GetCellData().AddArray(isovolumic_deformation_array)

    writer = vtk.vtkXMLUnstructuredGridWriter()
    writer.SetFileName(output_filepath)
    writer.SetInputData(output_compression)
    writer.Write()


def decomposition_to_vtu(input_filepath):
    print(f"Reading {input_filepath} ...")
    output_compression = read_vtu(input_filepath)
    centroid_arrays = get_centroid_arrays(output_compression)
    print(f"Extracted {len(centroid_arrays)} arrays from the vtu file")
    F = get_F_matrix(centroid_arrays)
    print(f"Extracted F matrix with shape {F.shape}")
    spherical_compression, isovolumic_deformation = decomposition(F)

    output_filepath = input_filepath.with_stem(input_filepath.stem + "_with_data")
    print(f"Saving output to {output_filepath} ...")
    add_data_to_vtu(
        output_compression,
        spherical_compression,
        isovolumic_deformation,
        output_filepath,
    )
    print("Done!")


def main():
    input_filepath = Path(
        "../../04_SIMULATIONS/445_R_93_F/C0003110_02/Step-Compression_26.vtu"
    )
    decomposition_to_vtu(input_filepath)


if __name__ == "__main__":
    main()
