from copy import copy
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pyssam
import SimpleITK as sitk
from scipy.interpolate import interp1d
from skimage.measure import find_contours
import pyvista as pv


def plot_cumulative_variance(explained_variance, target_variance=-1):
    number_of_components = np.arange(0, len(explained_variance)) + 1
    fig, ax = plt.subplots(1, 1)
    color = "blue"
    ax.plot(
        number_of_components,
        explained_variance * 100.0,
        marker="o",
        ms=2,
        color=color,
        mec=color,
        mfc=color,
    )
    if target_variance > 0.0:
        ax.axhline(target_variance * 100.0)

    ax.set_ylabel("Variance [%]")
    ax.set_xlabel("Number of components")
    ax.grid(axis="x")
    impath = Path(__file__).parent / "cumulative_variance.png"
    plt.savefig(impath, dpi=300)


def plot_shape_modes(
    ssm_obj,
    mean_shape_columnvector,
    mean_shape,
    original_shape_parameter_vector,
    shape_model_components,
    mode_to_plot,
):
    weights = [-2, 0, 2]
    fig, ax = plt.subplots(1, 3)
    for j, weights_i in enumerate(weights):
        shape_parameter_vector = copy(original_shape_parameter_vector)
        shape_parameter_vector[mode_to_plot] = weights_i
        mode_i_coords = ssm_obj.morph_model(
            mean_shape_columnvector, shape_model_components, shape_parameter_vector
        ).reshape(-1, 3)

        offset_dist = pyssam.utils.euclidean_distance(mean_shape, mode_i_coords)
        # colour points blue if closer to point cloud centre than mean shape
        mean_shape_dist_from_centre = pyssam.utils.euclidean_distance(
            mean_shape,
            np.zeros(3),
        )
        mode_i_dist_from_centre = pyssam.utils.euclidean_distance(
            mode_i_coords,
            np.zeros(3),
        )
        offset_dist = np.where(
            mode_i_dist_from_centre < mean_shape_dist_from_centre,
            offset_dist * -1,
            offset_dist,
        )
        if weights_i == 0:
            ax[j].scatter(
                mode_i_coords[:, 0],
                mode_i_coords[:, 2],
                c="gray",
                s=1,
            )
            ax[j].set_title("mean shape")
        else:
            ax[j].scatter(
                mode_i_coords[:, 0],
                mode_i_coords[:, 2],
                c=offset_dist,
                cmap="seismic",
                vmin=-1,
                vmax=1,
                s=1,
            )
            ax[j].set_title(f"mode {mode_to_plot} \nweight {weights_i}")
        ax[j].axis("off")
        ax[j].margins(0, 0)
        ax[j].xaxis.set_major_locator(plt.NullLocator())
        ax[j].yaxis.set_major_locator(plt.NullLocator())

    # plt.show()
    impath = Path(__file__).parent / "mode_plot.png"
    plt.savefig(impath, dpi=300)


def classify_and_store_contours(mask, slice_idx):
    contours = find_contours(mask[:, :, slice_idx])  # , level=0.5)
    if contours:
        outer_contours = contours[0]  # First contour as outer
        if len(contours) > 1:
            inner_contours = contours[1]  # Second contour as inner
    return outer_contours, inner_contours


def interpolate_contour(contour, num_points=100):
    distances = np.sqrt(np.sum(np.diff(contour, axis=0) ** 2, axis=1))
    cumulative_distances = np.insert(np.cumsum(distances), 0, 0)

    interp_func = interp1d(cumulative_distances, contour, axis=0, kind="linear")
    new_distances = np.linspace(0, cumulative_distances[-1], num_points)

    interpolated_contour = interp_func(new_distances)
    return interpolated_contour


def process_image(imnp):
    # get contour for each slice
    out_list = []
    inn_list = []
    for slice_idx in range(imnp.shape[2]):
        out, inn = classify_and_store_contours(imnp, slice_idx)
        # add slice_idx to array
        out = np.column_stack((out, np.ones(out.shape[0]) * slice_idx))
        inn = np.column_stack((inn, np.ones(inn.shape[0]) * slice_idx))

        # interpolate contours
        out = interpolate_contour(out, num_points=100)
        inn = interpolate_contour(inn, num_points=100)

        out_list.append(out)
        inn_list.append(inn)
    return out_list, inn_list


def get_paths(basepath: Path):
    # for all subdirs, get all '_CORTMASK' files
    #! analysing only tibiae now, hence filter 'T'
    #! assuming all subfolders containing tibiae are named 'T'
    cort_list = []
    for subdir in basepath.rglob('T'):
        if subdir.is_dir():
            for file in subdir.glob('*.mhd'):
                if file.is_file() and "_CORTMASK" in file.name and "mhd" in file.suffix:
                    cort_list.append(file)
    return cort_list


def extract_landmarks(cort_list: list):
    # Extract endosteal and periosteal surfaces
    all_points_endo = []
    all_points_peri = []
    for file in cort_list:
        print(f"Extracting contours from {file.name}")
        sitk_image = sitk.ReadImage(str(file))
        np_image = sitk.GetArrayFromImage(sitk_image)
        np_image = np.transpose(np_image, (2, 1, 0))
        np_image = np.flip(np_image, axis=0)
        np_image = np_image[:, :, 5:450]
        out_list, inn_list = process_image(np_image)
        out_list = np.array(out_list).reshape(-1, 3)
        inn_list = np.array(inn_list).reshape(-1, 3)

        all_points_endo.append(inn_list)
        all_points_peri.append(out_list)
    landmarks_peri = np.array(all_points_peri)
    landmarks_endo = np.array(all_points_endo)
    return landmarks_peri, landmarks_endo


def run_ssm():
    basepath = Path(
        "/home/simoneponcioni/Documents/01_PHD/03_Methods/HFE/01_DATA/TIBIA/flipped"
    )
    cort_list = get_paths(basepath)
    landmarks_peri, landmarks_endo = extract_landmarks(cort_list)
    for landmarks, compartment in zip(
        [landmarks_peri, landmarks_endo], ["periosteum", "endosteum"]
    ):
        ssm_obj = pyssam.SSM(landmarks)
        ssm_obj.create_pca_model(ssm_obj.landmarks_columns_scale)
        mean_shape_columnvector = ssm_obj.compute_dataset_mean()
        mean_shape = mean_shape_columnvector.reshape(-1, 3)
        shape_model_components = ssm_obj.pca_model_components
        print(shape_model_components)

        # Calculate average dimensions of the original shapes
        original_shapes = np.vstack(landmarks)
        avg_dimensions = np.mean(np.ptp(original_shapes, axis=0))

        # Scale the mean shape to match the average dimensions
        mean_shape_dimensions = np.ptp(mean_shape, axis=0)
        scaling_factors = avg_dimensions / mean_shape_dimensions
        scaled_mean_shape = mean_shape * scaling_factors
        # scaled_mean_shape[:, 2] = np.linspace(
        #     0, np.max(original_shapes[:, 2]), len(scaled_mean_shape[:, 2])
        # )
        print("test")

        print(
            f"To obtain {ssm_obj.desired_variance * 100}% variance, {ssm_obj.required_mode_number} modes are required"
        )
        plot_cumulative_variance(
            np.cumsum(ssm_obj.pca_object.explained_variance_ratio_), 0.9
        )

        mode_to_plot = 1
        print(
            f"explained variance is {ssm_obj.pca_object.explained_variance_ratio_[mode_to_plot]}"
        )

        plot_shape_modes(
            ssm_obj,
            mean_shape_columnvector,
            mean_shape,
            ssm_obj.model_parameters,
            ssm_obj.pca_model_components,
            mode_to_plot,
        )


        mean_shape_mesh = pv.PolyData(scaled_mean_shape)
        mean_shape_mesh.save(f"ssm_{compartment}.vtk")


def main():
    run_ssm()


if __name__ == "__main__":
    main()

