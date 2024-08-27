from pathlib import Path
import subprocess


def main():

    path_not_in_outfile = ['IMAGES/00000149/C3/T', 'IMAGES/00000150/C1/T', 'IMAGES/00000151/C1/T', 'IMAGES/00000174/C1/T', 'IMAGES/00000174/C2/T', 'IMAGES/00000174/C3/T', 'IMAGES/00000176/C1/R', 'IMAGES/00000185/C2/T', 'IMAGES/00000189/C1/R', 'IMAGES/00000190/C1/T', 'IMAGES/00000190/C2/T', 'IMAGES/00000190/C3/T', 'IMAGES/00000204/C1/T', 'IMAGES/00000233/C1/T', 'IMAGES/00000233/C2/T', 'IMAGES/00000269/C2/T']
    basepath = Path(__file__).parent
    
    scp_paths = []
    for sampledir in path_not_in_outfile:
        full_path = basepath / sampledir
        # glob and copy all files containing 'CORT_MASK' in the name
        for file in full_path.glob('*CORTMASK*'):
          scp_paths.append(str(f'{file.resolve()}\n'))
    with open('files_to_copy.txt', 'w') as file:
        file.writelines(scp_paths)


if __name__ == "__main__":
    main()
