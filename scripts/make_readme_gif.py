import os
import glob
from PIL import Image


def main():
    here = os.path.abspath(os.path.dirname(__file__))
    folder = os.path.abspath(os.path.join(here, '..', 'readme_qrs'))
    pngs = sorted(glob.glob(os.path.join(folder, '*.png')))
    if not pngs:
        raise SystemExit(f'No PNG files found in {folder}')

    images = [Image.open(p).convert('RGB') for p in pngs]

    # Normalize sizes (resize to the smallest w/h found) to avoid GIF issues
    min_w = min(img.width for img in images)
    min_h = min(img.height for img in images)
    if any((img.width, img.height) != (min_w, min_h) for img in images):
        images = [img.resize((min_w, min_h), Image.LANCZOS) for img in images]

    out_path = os.path.join(folder, 'examples.gif')
    images[0].save(
        out_path,
        save_all=True,
        append_images=images[1:],
        duration=1500,  # 1.5s per frame
        loop=0,
        disposal=2,
    )
    print(f'Wrote {out_path}')


if __name__ == '__main__':
    main()

