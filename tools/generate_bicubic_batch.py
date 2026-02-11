import argparse
from pathlib import Path

from PIL import Image


SUPPORTED_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Batch-generate bicubic baseline images.\n"
            "Default mode matches SRCNN test.py behavior: "
            "HR -> downsample by scale -> bicubic upsample."
        )
    )
    parser.add_argument("--input-dir", required=True, help="Input image directory")
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Output directory (default: same as input-dir)",
    )
    parser.add_argument("--scale", type=int, default=3, help="Upscale factor (>=2)")
    parser.add_argument(
        "--suffix",
        default=None,
        help="Output filename suffix (default: _bicubic_x{scale})",
    )
    parser.add_argument(
        "--mode",
        choices=["degrade_then_upsample", "upsample_only"],
        default="degrade_then_upsample",
        help=(
            "degrade_then_upsample: HR -> LR -> bicubic HR baseline; "
            "upsample_only: directly bicubic upscale input image."
        ),
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="Recursively scan input-dir",
    )
    parser.add_argument(
        "--include-generated",
        action="store_true",
        help="Also process files that already look like bicubic/srcnn outputs.",
    )
    parser.add_argument(
        "--keep-format",
        action="store_true",
        help="Keep original extension (default).",
    )
    parser.add_argument(
        "--output-format",
        choices=["png", "jpg", "bmp", "tiff", "webp"],
        default=None,
        help="Force output format/extension (e.g. png).",
    )
    return parser.parse_args()


def iter_images(input_dir: Path, recursive: bool):
    pattern = "**/*" if recursive else "*"
    for path in sorted(input_dir.glob(pattern)):
        if path.is_file() and path.suffix.lower() in SUPPORTED_EXTS:
            yield path


def looks_generated(path: Path):
    name = path.stem.lower()
    return "_bicubic_x" in name or "_srcnn_x" in name


def build_output_path(
    in_path: Path,
    input_root: Path,
    output_root: Path,
    suffix: str,
    output_format: str | None,
):
    rel = in_path.relative_to(input_root)
    rel_parent = rel.parent
    stem = in_path.stem
    if output_format:
        ext = "." + output_format.lower()
    else:
        ext = in_path.suffix
    return output_root / rel_parent / f"{stem}{suffix}{ext}"


def bicubic_generate(image: Image.Image, scale: int, mode: str):
    if mode == "upsample_only":
        return image.resize((image.width * scale, image.height * scale), Image.BICUBIC)

    # Match SRCNN-pytorch-master/test.py:
    # 1) align size to scale, 2) downsample, 3) bicubic upsample.
    aligned_w = (image.width // scale) * scale
    aligned_h = (image.height // scale) * scale
    if aligned_w <= 0 or aligned_h <= 0:
        raise ValueError(
            f"Image too small for scale={scale}: size=({image.width}, {image.height})"
        )
    img = image.resize((aligned_w, aligned_h), resample=Image.BICUBIC)
    img = img.resize((img.width // scale, img.height // scale), resample=Image.BICUBIC)
    img = img.resize((img.width * scale, img.height * scale), resample=Image.BICUBIC)
    return img


def main():
    args = parse_args()

    if args.scale < 2:
        raise ValueError("--scale must be >= 2")

    input_dir = Path(args.input_dir).resolve()
    if not input_dir.exists() or not input_dir.is_dir():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")

    output_dir = Path(args.output_dir).resolve() if args.output_dir else input_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    suffix = args.suffix if args.suffix else f"_bicubic_x{args.scale}"

    if args.output_format is not None and args.keep_format:
        print("Warning: --output-format is set, so --keep-format is ignored.")

    images = list(iter_images(input_dir, args.recursive))
    if not images:
        print(f"No images found in: {input_dir}")
        return

    success = 0
    failed = 0

    for in_path in images:
        try:
            if not args.include_generated and looks_generated(in_path):
                print(f"[SKIP] {in_path} (already generated-looking filename)")
                continue

            with Image.open(in_path) as im:
                rgb = im.convert("RGB")
                out_img = bicubic_generate(rgb, args.scale, args.mode)

            out_path = build_output_path(
                in_path=in_path,
                input_root=input_dir,
                output_root=output_dir,
                suffix=suffix,
                output_format=args.output_format,
            )
            out_path.parent.mkdir(parents=True, exist_ok=True)

            save_kwargs = {}
            if out_path.suffix.lower() in {".jpg", ".jpeg"}:
                save_kwargs = {"quality": 95}
            out_img.save(out_path, **save_kwargs)

            success += 1
            print(f"[OK] {in_path} -> {out_path}")
        except Exception as exc:
            failed += 1
            print(f"[FAILED] {in_path}: {exc}")

    print(f"\nDone. success={success}, failed={failed}, total={len(images)}")


if __name__ == "__main__":
    main()
