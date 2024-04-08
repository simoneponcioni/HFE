# distutils: language=c++
# cython: language_level=3

cimport numpy as np
cimport cython
from libc.math cimport rint, fmax, fmin
from cython.parallel import prange
import SimpleITK as sitk


ctypedef np.npy_intp SIZE_t

cdef class ROICalculator:
    cdef np.ndarray seg_array
    cdef np.ndarray mask_array
    cdef list spacing
    cdef float VOI_mm
    cdef float VOI
    cdef int seg_array_shape_0
    cdef int seg_array_shape_1
    cdef int seg_array_shape_2

    def __init__(self, np.ndarray seg_array, np.ndarray mask_array, list spacing, float VOI_mm, tuple seg_array_shape):
        self.seg_array = self.where_greater(seg_array, 0.1, 1, 0)
        self.mask_array = self.where_greater(mask_array, 0.1, 1, 0)
        self.spacing = spacing
        self.VOI_mm = VOI_mm
        self.VOI = VOI_mm / spacing[0]
        self.seg_array_shape_0, self.seg_array_shape_1, self.seg_array_shape_2 = seg_array_shape

    cdef np.ndarray where_greater(self, np.ndarray arr, float val, int x, int y):
        cdef SIZE_t i, n = arr.ndim
        cdef np.ndarray res = np.empty_like(arr)
        for i in prange(n, nogil=True):
            res[i] = x if arr[i] > val else y
        return res

    cdef tuple compute_roi_indices(self, tuple cog):
        cdef float x, y, z
        x, y, z = cog
        cdef int x_start, x_end, y_start, y_end, z_start, z_end
        x_start, x_end = int(rint(fmax(x - self.VOI / 2, 0))), int(rint(fmin(x + self.VOI / 2, self.seg_array_shape_0)))
        y_start, y_end = int(rint(fmax(y - self.VOI / 2, 0))), int(rint(fmin(y + self.VOI / 2, self.seg_array_shape_1)))
        z_start, z_end = int(rint(fmax(z - self.VOI / 2, 0))), int(rint(fmin(z + self.VOI / 2, self.seg_array_shape_2)))
        return x_start, x_end, y_start, y_end, z_start, z_end

    cdef float compute_bvtv(self, np.ndarray ROI_mask, np.ndarray ROI_mask_sphere, np.ndarray ROI_seg):
        cdef np.ndarray ROI_mask_sphere_mask = np.multiply(ROI_mask_sphere, ROI_mask)
        cdef np.ndarray ROI_sphere_seg = np.multiply(ROI_mask_sphere_mask, ROI_seg)
        cdef float intersection_percentage = np.sum(ROI_sphere_seg) / np.sum(ROI_mask_sphere)
        return fmax(0.01, fmin(1.0, intersection_percentage))

    cdef float calculate_for_cog(self, tuple cog):
        cdef int x_start, x_end, y_start, y_end, z_start, z_end
        x_start, x_end, y_start, y_end, z_start, z_end = self.compute_roi_indices(cog)
        cdef np.ndarray ROI_seg = self.seg_array[x_start:x_end, y_start:y_end, z_start:z_end]
        cdef np.ndarray ROI_mask = self.mask_array[x_start:x_end, y_start:y_end, z_start:z_end]
        cdef np.ndarray ROI_mask_sphere = self._generate_sphere_array(np.shape(ROI_seg), self.VOI / 2, [cog_i - start_i for cog_i, start_i in zip(cog, [x_start, y_start, z_start])])
        cdef float bvtv = self.compute_bvtv(ROI_mask, ROI_mask_sphere, ROI_seg)
        return bvtv

    cdef np.ndarray _generate_sphere_array(self, tuple shape, float radius, list position):
        cdef float semisizes[3]
        semisizes[:] = [radius] * 3
        cdef list grid = [slice(-x0, dim - x0) for x0, dim in zip(position, shape)]
        cdef np.ndarray position = np.ogrid[grid]
        cdef np.ndarray arr = np.zeros(np.asarray(shape).astype(int), dtype=float)
        cdef SIZE_t i, n = len(position)
        for i in prange(n, nogil=True):
            arr += np.abs(position[i] / semisizes[i]) ** 2
        return (arr <= 1.0).astype("int")

cdef np.ndarray load_img(str path):
    cdef sitk.Image img_sitk = sitk.ReadImage(path)
    cdef np.ndarray img_np = sitk.GetArrayFromImage(img_sitk)
    img_np = img_np.astype(np.uint8)
    img_np = img_np.transpose(2, 1, 0)
    img_np = np.flip(img_np, axis=0)
    return img_np

def main():
    cdef str seg_path = r"/Users/msb/Documents/01_PHD/03_Methods/HFE-ACCURATE/99_TEMP/compute_phi_rho_optimised/C0003111_UNCOMP_SEG.AIM_padded.mhd"
    cdef str mask_path = r"/Users/msb/Documents/01_PHD/03_Methods/HFE-ACCURATE/99_TEMP/compute_phi_rho_optimised/C0003111_TRAB_MASK_UNCOMP.AIM_padded.mhd"
    cdef np.ndarray seg_array = load_img(seg_path)
    cdef np.ndarray mask_array = load_img(mask_path)
    cdef list spacing = [0.061, 0.061, 0.061]
    cdef float VOI_mm = 1.3
    cdef ROICalculator roi_calculator = ROICalculator(seg_array, mask_array, spacing, VOI_mm, seg_array_shape)
    cdef np.ndarray cogs = np.random.rand(30000, 3) * np.array(seg_array.shape)
    cdef tuple cog
    cdef float bvtv
    for cog in cogs:
        bvtv = roi_calculator.calculate_for_cog(cog)

if __name__ == "__main__":
    main()