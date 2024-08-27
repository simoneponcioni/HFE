from pathlib import Path
import os


def cleanup_directories(rootdir):
    for file in rootdir.glob("**/*"):
        if (file.suffix in {".out", ".err"}):
            folder_name = file.stem.rpartition("_")[0]
            Path(folder_name).mkdir(exist_ok=True)
            file.rename(rootdir / folder_name / file.name)


def main():
    basepath = Path(__file__).parent
    cleanup_directories(basepath)
    

if __name__ == "__main__":
    main()
