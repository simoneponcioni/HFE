from pathlib import Path
import os
import json

def main():
    # Copy only the AIM files from the IMAGES directory to the RADIUS directory
    basedir = Path('/storage/workspaces/artorg_msb/hpc_abaqus/poncioni/HFE/00_ORIGAIM/REPRO/')

    # Create dictionary of names/directories correspondence (for hfe simulations.yaml config)
    names_dict = {}
    for file in basedir.rglob('*.AIM'):  # Use rglob to find files recursively
        names = file.stem.split('_')[0]
        names_dict[names] = str(file.parent.relative_to(basedir))  # Use file.parent.relative_to(basedir) to get the relative directory path
    names_dict = {k: names_dict[k] for k in sorted(names_dict)}

    # save it in a json
    outputdir = basedir / 'names_dict.json'
    with open(outputdir, 'w') as f:
        json.dump(names_dict, f, indent=4)

if __name__ == "__main__":
    main()
