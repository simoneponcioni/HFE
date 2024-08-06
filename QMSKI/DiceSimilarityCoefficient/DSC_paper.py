import json
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pyvista as pv
import SimpleITK as sitk
import vtk
from vtk.util.numpy_support import numpy_to_vtk, vtk_to_numpy

# flake8: noqa: E501


class DiceSimilarityCoefficient:
    def __init__(
        self,
        cortmask_path: Path,
        trabmask_path: Path,
        mesh_path: Path,
        show_plots=False,
    ):
        self.cortmask_path = str(cortmask_path.resolve())
        self.trabmask_path = str(trabmask_path.resolve())
        self.mesh_path = str(mesh_path.resolve())
        self.show_plots = show_plots

    def polydata_to_imagedata(self, polydata, dimensions=(100, 100, 100), padding=1):
        """
        https://github.com/tfmoraes/polydata_to_imagedata/blob/main/polydata_to_imagedata.py
        """
        xi, xf, yi, yf, zi, zf = polydata.GetBounds()
        dx, dy, dz = dimensions

        # Calculating spacing
        sx = (xf - xi) / dx
        sy = (yf - yi) / dy
        sz = (zf - zi) / dz

        # Calculating Origin
        ox = xi + sx / 2.0
        oy = yi + sy / 2.0
        oz = zi + sz / 2.0

        if padding:
            ox -= sx
            oy -= sy
            oz -= sz

            dx += 2 * padding
            dy += 2 * padding
            dz += 2 * padding

        image = vtk.vtkImageData()
        image.SetSpacing((sx, sy, sz))
        image.SetDimensions((dx, dy, dz))
        image.SetExtent(0, dx - 1, 0, dy - 1, 0, dz - 1)
        image.SetOrigin((ox, oy, oz))
        image.AllocateScalars(vtk.VTK_UNSIGNED_CHAR, 1)

        inval = 1
        outval = 0

        for i in range(image.GetNumberOfPoints()):
            image.GetPointData().GetScalars().SetTuple1(i, inval)

        pol2stenc = vtk.vtkPolyDataToImageStencil()
        pol2stenc.SetInputData(polydata)
        pol2stenc.SetOutputOrigin((ox, oy, oz))
        pol2stenc.SetOutputSpacing((sx, sy, sz))
        pol2stenc.SetOutputWholeExtent(image.GetExtent())
        pol2stenc.Update()

        imgstenc = vtk.vtkImageStencil()
        imgstenc.SetInputData(image)
        imgstenc.SetStencilConnection(pol2stenc.GetOutputPort())
        imgstenc.ReverseStencilOff()
        imgstenc.SetBackgroundValue(outval)
        imgstenc.Update()

        return imgstenc.GetOutput()

    def vtk2numpy(self, imvtk):
        """turns a vtk image data into a numpy array"""
        dim = imvtk.GetDimensions()
        data = imvtk.GetPointData().GetScalars()
        imnp = vtk_to_numpy(data)
        # vtk and numpy have different array conventions
        imnp = imnp.reshape(dim[2], dim[1], dim[0])
        imnp = imnp.transpose(2, 1, 0)
        return imnp

    def dice_coefficient(self, truth, prediction):
        intersection = np.logical_and(truth, prediction)
        return 1 - 2.0 * intersection.sum() / (truth.sum() + prediction.sum())

    def compute(self):
        # read masks and sum them together
        cort_mask = sitk.ReadImage(self.cortmask_path)
        trab_mask = sitk.ReadImage(self.trabmask_path)
        mask_sitk = cort_mask + trab_mask

        mask_np = sitk.GetArrayFromImage(mask_sitk)
        mask_np = np.flip(mask_np, axis=1)

        if self.show_plots:
            MIDSLICE = mask_np.shape[2] // 2
            plt.figure()
            plt.imshow(mask_np[:, :, MIDSLICE], cmap="gray")
            parent_path = os.path.dirname(self.cortmask_path)
            file_name = os.path.basename(self.cortmask_path)
            img_name = os.path.join(parent_path, f"{file_name}_mask_np.png")
            plt.savefig(img_name, dpi=100)
            plt.close()

        mesh = pv.read(self.mesh_path)
        mesh["density"] = np.full(mesh.n_cells, 1)
        mesh_grid = pv.create_grid(
            mesh, dimensions=(mask_np.shape[2], mask_np.shape[1], mask_np.shape[0])
        )
        mesh_res = mesh_grid.sample(mesh)

        density_data = mesh_res.get_array(name="density")
        density_data_3d = density_data.reshape(
            [mask_np.shape[0], mask_np.shape[1], mask_np.shape[2]]
        )

        mask_np = np.where(mask_np > 0, 1, 0)
        density_data_3d = np.where(density_data_3d > 0, 1, 0)

        if self.show_plots:
            MIDSLICE = mask_np.shape[2] // 2
            plt.figure()
            plt.imshow(mask_np[:, :, MIDSLICE], cmap="gray")
            plt.imshow(density_data_3d[:, :, MIDSLICE], cmap="gray", alpha=0.5)
            plt.colorbar()
            # plt.show()
            parent_path = os.path.dirname(self.cortmask_path)
            file_name = os.path.basename(self.cortmask_path)
            img_name = os.path.join(parent_path, f"{file_name}_density_np.png")
            plt.savefig(img_name, dpi=100)
            plt.close()

        mask_sitk = sitk.GetImageFromArray(mask_np)
        mesh_sitk = sitk.GetImageFromArray(density_data_3d)
        mask_sitk.CopyInformation(mask_sitk)
        mesh_sitk.CopyInformation(mask_sitk)
        
        # save images to check overlap
        parent_path = os.path.dirname(self.cortmask_path)
        file_name = os.path.splitext(os.path.basename(self.cortmask_path))[0]
        img_name = os.path.join(parent_path, f"{file_name}_mask_np.png")
        sitk.WriteImage(mask_sitk, f'{img_name}_mask_dsc.mha')
        sitk.WriteImage(mesh_sitk, f'{img_name}_mesh_dsc.mha')

        meas = sitk.LabelOverlapMeasuresImageFilter()
        meas.Execute(mask_sitk, mesh_sitk)
        dice = meas.GetDiceCoefficient()
        print(dice)
        return dice


def main():
    basepath = Path(
        "/home/simoneponcioni/Documents/01_PHD/04_Output-Reports-Presentations-Publications/HFE-RESULTS/DSC_PAPER/TIBIA"
    )

    cortmasks_list = list(basepath.glob("*_CORTMASK.mhd"))

    dice_results = {}
    for cortmask in cortmasks_list:
        trabmask = cortmask.parent / (
            cortmask.stem.replace("CORTMASK", "TRABMASK") + ".mhd"
        )
        mesh = cortmask.parent / (cortmask.stem.replace("CORTMASK", "MESH") + ".vtu")

        name = cortmask.stem.split("_")[0]

        print(f"Processing {name}:\n\t{cortmask}\n\t{trabmask}\n\t{mesh}")
        dsc = DiceSimilarityCoefficient(cortmask, trabmask, mesh, show_plots=True)
        coeff = dsc.compute()
        print(f"{name}: {coeff}")
        dice_results[name] = coeff

    print(dice_results)
    with open("dice_results.json", "w") as f:
        json.dump(dice_results, f)

    # stats: calculate mean and std of dice_results
    dice_values = list(dice_results.values())
    mean_dice = np.mean(dice_values)
    std_dice = np.std(dice_values)
    print(f"Mean dice: {mean_dice}")
    print(f"Std dice: {std_dice}")


if __name__ == "__main__":
    main()
