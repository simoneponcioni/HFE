import os
from pathlib import Path
import json

# flake8: noqa: E501


class Odb2VtkWrapper:
    def __init__(self, odb2vtk_path, odb_path, cfg, only_last_frame=False):
        self.odb2vtk_path = odb2vtk_path
        self.odb_path = Path(odb_path)
        self.vtk_path = self.odb_path.with_suffix(".vtk")
        # self.abq_path = cfg.solver.abaqus
        self.abq_path = "/var/DassaultSystemes/SIMULIA/Commands/abq2021hf6"
        self.only_last_frame = only_last_frame

    def get_json(self):
        def json2dict(json_path):
            with open(json_path, "r") as f:
                return json.load(f)

        os.system(
            f"{self.abq_path} python {self.odb2vtk_path} --header 1 --odbFile {self.odb_path}"
        )
        json_path = self.odb_path.with_suffix(".json")
        print(f"JSON written to {json_path}")
        return json2dict(json_path)

    def convert(self):
        json_dict = self.get_json()
        instance = json_dict["instances"][0]
        instance_str = f'"{instance}"'

        steps = json_dict["steps"]
        stepname = steps[0][0]
        frames = [int(step.split("-frame-")[-1]) for step in steps[0][1]]
        if self.only_last_frame:
            frames = [frames[-1]]
        step_cli = f"{stepname}:{','.join(map(str, frames))}"

        print(stepname)
        print(step_cli)

        os.system(
            f"{self.abq_path} python {self.odb2vtk_path} --header 0 --instance {instance_str} --step {step_cli} --odbFile {self.odb_path}"
        )
        return self.vtk_path


def test():
    cfg = {"solver": {"abaqus": "/var/DassaultSystemes/SIMULIA/Commands/abq2021hf6"}}
    odb2vtkpath = "/home/simoneponcioni/Documents/04_TOOLS/ODB2VTK/python/odb2vtk.py"
    odb_path = "/home/simoneponcioni/Desktop/odb2vtk_tests/tibia_test/test_tibia.odb"

    wrapper = Odb2VtkWrapper(odb2vtkpath, odb_path, cfg, only_last_frame=True)
    vtk_path = wrapper.convert()
    print(vtk_path)


if __name__ == "__main__":
    test()
