from pathlib import Path
import vtk
import numpy as np


def read_vtu(filepath):
    stepCompression_6vtu = vtk.vtkXMLUnstructuredGridReader()
    stepCompression_6vtu.SetFileName(str(filepath))
    stepCompression_6vtu.Update()
    output_compression = stepCompression_6vtu.GetOutput()
    return output_compression


def decomposition(F):
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
    arrays = output_compression.GetCellData()
    centroid_arrays = {
        arrays.GetArrayName(i): arrays.GetArray(i)
        for i in range(arrays.GetNumberOfArrays())
        if "Centroid" in arrays.GetArrayName(i)
    }
    return centroid_arrays


def get_F_matrix(centroid_arrays):
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
    output_compression, spherical_compression, isovolumic_deformation, output_filepath
):
    # Create new arrays for spherical compression and isovolumic deformation
    spherical_compression_array = vtk.vtkFloatArray()
    spherical_compression_array.SetName("SphericalCompression")
    isovolumic_deformation_array = vtk.vtkFloatArray()
    isovolumic_deformation_array.SetName("IsovolumicDeformation")

    # Set the values of the arrays
    for i in range(len(spherical_compression)):
        spherical_compression_array.InsertNextValue(spherical_compression[i])
        isovolumic_deformation_array.InsertNextValue(isovolumic_deformation[i])

    # Add the arrays to the output_compression vtk unstructured grid
    output_compression.GetCellData().AddArray(spherical_compression_array)
    output_compression.GetCellData().AddArray(isovolumic_deformation_array)

    # Save the vtk unstructured grid with all the cell data
    writer = vtk.vtkXMLUnstructuredGridWriter()
    writer.SetFileName(output_filepath)
    writer.SetInputData(output_compression)
    writer.Write()


def main():
    input_filepath = Path(
        "/home/simoneponcioni/Documents/01_PHD/03_Methods/HFE/99_TEMP/damage_maps/C0003091_02/Step-Compression_4.vtu"
    )
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


if __name__ == "__main__":
    main()
