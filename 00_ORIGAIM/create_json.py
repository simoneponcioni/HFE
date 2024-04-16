from pathlib import Path
import os
import json

def main():
    # Copy only the AIM files from the IMAGES directory to the RADIUS directory
    basedir = Path('/home/sp20q110/HFE/00_ORIGAIM/RADIUS')
    # outputdir = Path('/home/simoneponcioni/Documents/radius/RADIUS')
    # outputdir.mkdir(parents=True, exist_ok=True)

    # for subdir in basedir.iterdir():
    #     output_subdir = outputdir / subdir.name
    #     output_subdir.mkdir(parents=True, exist_ok=True)
    #     for file in subdir.iterdir():
    #         if file.suffix == '.AIM':
    #             os.system(f'cp {file} {output_subdir}')

    # Create dictionary of names/directories correspondence (for hfe simulations.yaml config)
    names_dict = {}
    for subdir in basedir.iterdir():
        try:
            for file in subdir.iterdir():
                if file.suffix == '.AIM':
                    names = file.stem.split('_')[0]
                    names_dict[names] = subdir.stem
        except NotADirectoryError:
            pass
    names_dict = {k: names_dict[k] for k in sorted(names_dict)}
            
    # save it in a json
    outputdir = basedir / 'names_dict.json'
    with open(outputdir, 'w') as f:
        json.dump(names_dict, f, indent=4)
        
if __name__ == "__main__":
    main()