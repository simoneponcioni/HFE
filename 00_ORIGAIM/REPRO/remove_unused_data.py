from pathlib import Path

def main():
    basepath = Path('IMAGES')
    for file in basepath.glob('**/*'):
        if '.GOBJ' in file.suffix:
            file.unlink()
            
if __name__ == "__main__":
    main()