import gmsh
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import pandas as pd


class SignedInverseCohomologyNumber:
    def __init__(self, meshpath: Path, show_plots=False):
        self.meshpath = meshpath
        self.show_plots = show_plots

    def compute(self):
        mesh_name = self.meshpath.name
        print(f"Processing mesh: {mesh_name}")
        gmsh.initialize()
        gmsh.merge(str(self.meshpath.resolve()))
        gmsh.model.occ.synchronize()

        # get element qualities:
        # _, eleTags, _ = gmsh.model.mesh.getElements(dim=3)
        # q = gmsh.model.mesh.getElementQualities(eleTags[0], "minSICN")

        # alternative using plugin:
        gmsh.plugin.setNumber("AnalyseMeshQuality", "ICNMeasure", 1.0)
        gmsh.plugin.setNumber("AnalyseMeshQuality", "CreateView", 1.0)
        t = gmsh.plugin.run("AnalyseMeshQuality")
        _, _, data, _, _ = gmsh.view.getModelData(t, 0)
        gmsh.finalize()

        if self.show_plots:
            plt.figure(figsize=(10, 5))
            plt.hist(np.array(data).flatten(), bins=100, color="tab:blue")
            plt.title(f"{mesh_name.split('s')[0]}: (S-) ICN", weight="bold")
            plt.show()

        sicn_avg = np.mean(data)
        sicn_25 = np.percentile(data, 25)
        sicn_75 = np.percentile(data, 75)
        return sicn_avg, sicn_25, sicn_75


def main():
    basepath = Path(
        "/home/simoneponcioni/Documents/01_PHD/04_Output-Reports-Presentations-Publications/HFE-RESULTS/SICN_PAPER/"
    )
    # get all the .msh files into a list also in subdirectories
    msh_files = list(basepath.rglob("*.msh"))

    sicn_dict = {}
    for meshpath in msh_files:
        sicn = SignedInverseCohomologyNumber(meshpath, show_plots=False)
        sicn_avg, sicn_25, sicn_75 = sicn.compute()
        print(f"Average (S-)ICN: {sicn_avg}")
        print(f"\t25th percentile: {sicn_25}")
        print(f"\t75th percentile: {sicn_75}")
        sicn_dict[meshpath.name] = [sicn_avg, sicn_25, sicn_75]

    # make it a pandas dataframe
    df = pd.DataFrame.from_dict(
        sicn_dict, orient="index", columns=["avg", "25th", "75th"]
    )
    df.to_csv(basepath / "sicn_results.csv")

    # average of the results for each column
    avg_avg = df["avg"].mean()
    avg_25 = df["25th"].mean()
    avg_75 = df["75th"].mean()

    print("Average of the average values: ", avg_avg)
    print("Average of the 25th percentile values: ", avg_25)
    print("Average of the 75th percentile values: ", avg_75)


if __name__ == "__main__":
    main()
