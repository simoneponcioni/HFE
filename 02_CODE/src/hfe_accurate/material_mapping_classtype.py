import cProfile
import logging
import pickle
from time import time
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import SimpleITK as sitk

# flake8: noqa: E501

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)


class ROICalculator:
    def __init__(
        self,
        seg_array: np.ndarray,
        mask_array: np.ndarray,
        spacing: List[float],
        VOI_mm: float,
    ):
        self.seg_array = np.where(seg_array > 0.1, 1, 0)
        self.mask_array = np.where(mask_array > 0.1, 1, 0)
        self.spacing = spacing
        self.VOI = VOI_mm / spacing[0]

    def compute_roi_indices(self, cog):
        x, y, z = cog
        x_start, x_end = int(np.rint(max(x - self.VOI / 2, 0))), int(
            np.rint(min(x + self.VOI / 2, self.seg_array.shape[0]))
        )
        y_start, y_end = int(np.rint(max(y - self.VOI / 2, 0))), int(
            np.rint(min(y + self.VOI / 2, self.seg_array.shape[1]))
        )
        z_start, z_end = int(np.rint(max(z - self.VOI / 2, 0))), int(
            np.rint(min(z + self.VOI / 2, self.seg_array.shape[2]))
        )
        return x_start, x_end, y_start, y_end, z_start, z_end

    def compute_phi_(self, ROI_mask):
        try:
            phi = float(np.count_nonzero(ROI_mask)) / float(np.size(ROI_mask))
        except ZeroDivisionError:
            phi = 0.0

        if np.isnan(phi):
            phi = 0.0
        if phi > 1.0:
            phi = 1.0
        return phi

    def compute_bvtv(self, ROI_mask, ROI_mask_sphere, ROI_seg):
        ROI_mask_sphere_mask = np.multiply(ROI_mask_sphere, ROI_mask)
        ROI_sphere_seg = np.multiply(ROI_mask_sphere_mask, ROI_seg)

        intersection_percentage = np.sum(ROI_sphere_seg) / np.sum(ROI_mask_sphere)
        return max(0.01, min(1.0, intersection_percentage))

    def calculate_for_cog(self, cog):
        x_start, x_end, y_start, y_end, z_start, z_end = self.compute_roi_indices(cog)

        ROI_seg = self.seg_array[x_start:x_end, y_start:y_end, z_start:z_end]
        ROI_mask = self.mask_array[
            x_start:x_end,
            y_start:y_end,
            z_start:z_end,
        ]
        pbv = self.compute_phi_(ROI_mask)
        if pbv > 0.0:
            ROI_mask_sphere = self._generate_sphere_array(
                shape=np.shape(ROI_seg),
                radius=self.VOI / 2,
                position=[
                    cog_i - start_i
                    for cog_i, start_i in zip(cog, [x_start, y_start, z_start])
                ],
            )

            bvtv = self.compute_bvtv(ROI_mask, ROI_mask_sphere, ROI_seg)
        else:
            bvtv = 0.0
        return pbv, bvtv

    def _generate_sphere_array(self, shape, radius, position):
        semisizes = (float(radius),) * 3
        grid = [slice(-x0, dim - x0) for x0, dim in zip(position, shape)]
        position = np.ogrid[grid]
        arr = np.zeros(np.asarray(shape).astype(int), dtype=float)
        for x_i, semisize in zip(position, semisizes):
            arr += np.abs(x_i / semisize) ** 2
        return (arr <= 1.0).astype("int")


def load_img(path):
    img_sitk = sitk.ReadImage(path)
    img_np = sitk.GetArrayFromImage(img_sitk)
    img_np = img_np.astype(np.uint8)
    # img_np = img_np.transpose(2, 1, 0)
    # img_np = np.flip(img_np, axis=0)
    return img_np


def main():
    seg_path = r"/Users/msb/Documents/01_PHD/03_Methods/HFE-ACCURATE/99_TEMP/compute_phi_rho_optimised/C0003111_UNCOMP_SEG_padded.mhd"
    mask_path_cort = r"/Users/msb/Documents/01_PHD/03_Methods/HFE-ACCURATE/99_TEMP/compute_phi_rho_optimised/C0003111_CORT_MASK_UNCOMP_padded.mhd"
    mask_path_trab = r"/Users/msb/Documents/01_PHD/03_Methods/HFE-ACCURATE/99_TEMP/compute_phi_rho_optimised/C0003111_TRAB_MASK_UNCOMP_padded.mhd"
    seg_array = load_img(seg_path)
    mask_array_cort = load_img(mask_path_cort)
    mask_array_trab = load_img(mask_path_trab)
    mask_arrays = [mask_array_cort, mask_array_trab]
    SEG_correction = True

    spacing = [0.061, 0.061, 0.061]
    VOI_cort_mm = 1.3
    VOI_trab_mm = 4.0
    VOIS = [VOI_cort_mm, VOI_trab_mm]

    # create an array of 30000 3D coordinates
    cogs = np.random.rand(30000, 3) * np.array(seg_array.shape)

    cogs_cort_dict = pickle.load(open("centroids_cort_C0003111.pkl", "rb"))
    cogs_trab_dict = pickle.load(open("centroids_trab_C0003111.pkl", "rb"))
    cogs_cort = cogs_cort_dict.values()
    cogs_trab = cogs_trab_dict.values()
    # divide the cogs by spacing with numpy operation
    cogs_cort = np.round(np.array(list(cogs_cort)) / spacing).astype(int)
    cogs_trab = np.round(np.array(list(cogs_trab)) / spacing).astype(int)
    print(cogs_cort.shape)
    print(cogs_trab.shape)

    cogs_list = [cogs_cort, cogs_trab]
    compartment_names = ["cogs_cort", "cogs_trab"]
    for cogs, compartment_s, mask_array, VOI_mm in zip(
        cogs_list, compartment_names, mask_arrays, VOIS
    ):
        timestart = time()
        roi_calculator = ROICalculator(seg_array, mask_array, spacing, VOI_mm)
        phi = np.zeros(len(cogs))
        rho = np.zeros(len(cogs))
        rho_fe = np.zeros(len(cogs))
        for i, cog in enumerate(cogs):
            phi_s, rho_s = roi_calculator.calculate_for_cog(cog)

            # if SEG_correction == True:
            #     TOL_SPACING = 0.01
            #     if abs(spacing[0] - 0.061) <= TOL_SPACING:
            #         # Correction curve from Varga et al. 2009 for XCTII
            #         rho_s = rho_s * 0.651 + 0.056462
            #     elif abs(spacing[0] - 0.082) <= TOL_SPACING:
            #         # Correction curve from Varga et al. 2009 for XCTI, added by MI
            #         rho_s = rho_s * 0.745745 - 0.0209902
            #     else:
            #         raise ValueError(
            #             f"SEG_correction is True but 'spacing' is not 0.061 nor 0.082 (it is {spacing[0]}))"
            #         )
            # else:
            #     pass
            phi[i] = phi_s
            rho[i] = rho_s
            rho_fe[i] = 1  # ! adapt this

        timeend = time()
        elaps_time = timeend - timestart
        logger.info(
            f"Elapsed time for {compartment_s} compartment: {elaps_time:.3f} seconds"
        )

        fig, axs = plt.subplots(2)
        axs[0].hist(phi.flatten(), bins=100)
        axs[0].set_title("PBV")
        axs[1].hist(rho.flatten(), bins=100)
        axs[1].set_title("BV/TV")

        plt.tight_layout()
        plt.savefig(f"hist_{compartment_s}.png")


if __name__ == "__main__":
    main()
