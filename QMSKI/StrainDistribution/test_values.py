from pathlib import Path
import pyvista as pv
import numpy as np
import matplotlib.colors as colors
import matplotlib.cm as cm


def plot_midslice_vtk(vtkpath: Path, celldata: str):
    def custom_colormap():
        # Create a custom colormap
        cmap = colors.LinearSegmentedColormap.from_list(
            "transparent_to_red", [(0, 0, 0, 0), (1, 0, 0, 1)]
        )

        norm = colors.Normalize(
            vmin=np.min(clip_masked[celldata]), vmax=np.max(clip_masked[celldata])
        )
        color_mapper = cm.ScalarMappable(norm=norm, cmap=cmap)
        return color_mapper, cmap

    vtkpv = pv.read(vtkpath)
    sliced = vtkpv.clip(normal=[1, 0, 0], invert=True)

    clip_masked = sliced.threshold(
        [
            np.quantile(sliced.cell_data[celldata], 0.00),
            np.quantile(sliced.cell_data[celldata], 0.98),
        ],
        scalars=celldata,
        invert=False,
    )

    # color_mapper, cmap = custom_colormap()
    # clip_masked['IsovolumicDeformation_Colors'] = color_mapper.to_rgba(clip_masked[celldata])

    edges = sliced.extract_feature_edges(60)

    p = pv.Plotter(notebook=False, off_screen=True)
    p.add_mesh(edges, color="black", lighting=False)
    p.add_mesh(
        clip_masked,
        # scalars='IsovolumicDeformation_Colors',
        scalars="IsovolumicDeformation",
        # cmap=cmap,
        cmap="PuBu",
        show_edges=False,
        log_scale=False,
        scalar_bar_args={"title": celldata, "n_labels": 5},
    )
    p.view_yz()
    p.camera.roll += 180
    p.camera.SetParallelProjection(True)
    p.camera.zoom(1.4)
    p.show()
    p.screenshot(__file__ + ".png")

    # name_with_celldata = vtkpath.stem + '_' + celldata
    # vtk_parent_path = vtkpath.parent / 'midslice'
    # vtk_parent_path.mkdir(parents=True, exist_ok=True)
    # p.screenshot(vtk_parent_path / f'{name_with_celldata}.png')


vtudir = r"/home/simoneponcioni/Documents/01_PHD/04_Output-Reports-Presentations-Publications/HFE-RESULTS/strain-distribution/20240717/436_C0003113.vtu"
# patlib glob through all the vtu files

celldata_iso = "IsovolumicDeformation"
celldata_sph = "SphericalCompression"
celldata_list = [celldata_iso, celldata_sph]


def main():
    plot_midslice_vtk(Path(vtudir), celldata_iso)


if __name__ == "__main__":
    main()
