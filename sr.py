import argparse
import os

import numpy as np
from PIL import Image, ImageFilter
try:
    import cv2
except Exception:
    cv2 = None
try:
    import torch
    from torch import nn
except Exception:
    torch = None
    nn = None

_SRCNN_CACHE = {"model": None, "weights": None, "device": None}


def _resize_np(arr, size, resample):
    channels = []
    for c in range(arr.shape[2]):
        chan = Image.fromarray(arr[:, :, c], mode="F").resize(size, resample)
        channels.append(np.asarray(chan, dtype=np.float32))
    return np.stack(channels, axis=2)


def _convert_rgb_to_ycbcr(img):
    y = 16. + (64.738 * img[:, :, 0] + 129.057 * img[:, :, 1] + 25.064 * img[:, :, 2]) / 256.
    cb = 128. + (-37.945 * img[:, :, 0] - 74.494 * img[:, :, 1] + 112.439 * img[:, :, 2]) / 256.
    cr = 128. + (112.439 * img[:, :, 0] - 94.154 * img[:, :, 1] - 18.285 * img[:, :, 2]) / 256.
    return np.array([y, cb, cr]).transpose([1, 2, 0])


def _convert_ycbcr_to_rgb(img):
    r = 298.082 * img[:, :, 0] / 256. + 408.583 * img[:, :, 2] / 256. - 222.921
    g = 298.082 * img[:, :, 0] / 256. - 100.291 * img[:, :, 1] / 256. - 208.120 * img[:, :, 2] / 256. + 135.576
    b = 298.082 * img[:, :, 0] / 256. + 516.412 * img[:, :, 1] / 256. - 276.836
    return np.array([r, g, b]).transpose([1, 2, 0])


if nn is not None:
    class _SRCNN(nn.Module):
        def __init__(self, num_channels=1):
            super().__init__()
            self.conv1 = nn.Conv2d(num_channels, 64, kernel_size=9, padding=9 // 2)
            self.conv2 = nn.Conv2d(64, 32, kernel_size=5, padding=5 // 2)
            self.conv3 = nn.Conv2d(32, num_channels, kernel_size=5, padding=5 // 2)
            self.relu = nn.ReLU(inplace=True)

        def forward(self, x):
            x = self.relu(self.conv1(x))
            x = self.relu(self.conv2(x))
            x = self.conv3(x)
            return x
else:
    class _SRCNN:
        def __init__(self, *args, **kwargs):
            raise ImportError("PyTorch is required for SRCNN mode. Install torch to use this feature.")


def _load_srcnn(weights_path, device):
    if torch is None or nn is None:
        raise ImportError("PyTorch is required for SRCNN mode. Install torch to use this feature.")
    if not os.path.exists(weights_path):
        raise FileNotFoundError(f"Weights file not found: {weights_path}")

    cached = _SRCNN_CACHE
    if cached["model"] is not None and cached["weights"] == weights_path and cached["device"] == device:
        return cached["model"]

    model = _SRCNN().to(device)
    state_dict = model.state_dict()
    weights = torch.load(weights_path, map_location="cpu")
    for name, param in weights.items():
        if name in state_dict:
            state_dict[name].copy_(param)
        else:
            raise KeyError(f"Unexpected key in weights: {name}")

    model.eval()
    cached["model"] = model
    cached["weights"] = weights_path
    cached["device"] = device
    return model


def unsharp_mask(img, radius=1.2, amount=0.8, threshold=0.0, luminance_only=False):
    if luminance_only and img.mode == "RGB":
        y, cb, cr = img.convert("YCbCr").split()
        y_sharp = unsharp_mask(y, radius=radius, amount=amount, threshold=threshold)
        return Image.merge("YCbCr", (y_sharp, cb, cr)).convert("RGB")

    blurred = img.filter(ImageFilter.GaussianBlur(radius))
    arr = np.asarray(img, dtype=np.float32)
    blur = np.asarray(blurred, dtype=np.float32)
    mask = arr - blur
    if threshold > 0:
        mask = np.where(np.abs(mask) < threshold, 0.0, mask)
    sharp = arr + amount * mask
    sharp = np.clip(sharp, 0, 255).astype(np.uint8)
    return Image.fromarray(sharp)


def enhance_detail_cv(img, sharpen_amount=1.0, sharpen_radius=1.2, detail_boost=0.5):
    if cv2 is None:
        # Fallback to PIL unsharp mask if OpenCV is unavailable
        out = img
        if sharpen_amount > 0:
            out = unsharp_mask(out, radius=sharpen_radius, amount=sharpen_amount, luminance_only=True)
        if detail_boost > 0:
            out = unsharp_mask(
                out,
                radius=max(2.0, sharpen_radius * 2.5),
                amount=detail_boost,
                threshold=2.0,
                luminance_only=True,
            )
        return out

    bgr = cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)
    orig = bgr.copy()
    detail_boost = max(0.0, float(detail_boost))

    if detail_boost > 0:
        lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
        l_chan, a_chan, b_chan = cv2.split(lab)
        clip = 1.2 + detail_boost * 2.0
        clahe = cv2.createCLAHE(clipLimit=clip, tileGridSize=(8, 8))
        l_chan = clahe.apply(l_chan)
        lab = cv2.merge((l_chan, a_chan, b_chan))
        bgr = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

    if sharpen_amount > 0:
        sigma = max(0.6, sharpen_radius)
        blur = cv2.GaussianBlur(bgr, (0, 0), sigma)
        amt = min(sharpen_amount, 1.2) * 0.85
        bgr = cv2.addWeighted(bgr, 1.0 + amt, blur, -amt, 0)

    if detail_boost > 0:
        sigma2 = max(1.0, sharpen_radius * 1.4)
        blur2 = cv2.GaussianBlur(bgr, (0, 0), sigma2)
        amt2 = detail_boost * 0.5
        bgr = cv2.addWeighted(bgr, 1.0 + amt2, blur2, -amt2, 0)

    bgr = cv2.addWeighted(bgr, 0.85, orig, 0.15, 0)

    bgr = np.clip(bgr, 0, 255).astype(np.uint8)
    return Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))


def iterative_back_projection(lr_img, init_hr_img, scale=2, iterations=8, backproj_lambda=1.0):
    lr_arr = np.asarray(lr_img, dtype=np.float32)
    hr_arr = np.asarray(init_hr_img, dtype=np.float32)

    lr_size = (lr_arr.shape[1], lr_arr.shape[0])
    hr_size = (lr_size[0] * scale, lr_size[1] * scale)

    if (hr_arr.shape[1], hr_arr.shape[0]) != hr_size:
        hr_arr = _resize_np(hr_arr, hr_size, Image.BICUBIC)

    for _ in range(iterations):
        down = _resize_np(hr_arr, lr_size, Image.BICUBIC)
        err = lr_arr - down
        err_up = _resize_np(err, hr_size, Image.BICUBIC)
        hr_arr = hr_arr + backproj_lambda * err_up
        hr_arr = np.clip(hr_arr, 0, 255)

    return Image.fromarray(hr_arr.astype(np.uint8))


def super_resolve(
    lr_img,
    scale=2,
    sharpen_amount=0.9,
    sharpen_radius=1.3,
    detail_boost=0.25,
    ibp_iters=10,
    ibp_lambda=1.0,
):
    if scale < 2:
        raise ValueError("scale must be >= 2")

    hr = lr_img.resize((lr_img.width * scale, lr_img.height * scale), Image.LANCZOS)

    if ibp_iters > 0:
        hr = iterative_back_projection(
            lr_img,
            hr,
            scale=scale,
            iterations=ibp_iters,
            backproj_lambda=ibp_lambda,
        )

    hr = enhance_detail_cv(
        hr,
        sharpen_amount=sharpen_amount,
        sharpen_radius=sharpen_radius,
        detail_boost=detail_boost,
    )

    return hr


def super_resolve_srcnn(lr_img, scale=2, weights_path=None, device="auto"):
    if scale < 2:
        raise ValueError("scale must be >= 2 for SRCNN mode.")
    if not weights_path:
        raise ValueError("weights_path is required for SRCNN mode.")

    if torch is None:
        raise ImportError("PyTorch is required for SRCNN mode. Install torch to use this feature.")

    if device == "auto":
        device = "cuda:0" if torch.cuda.is_available() else "cpu"

    model = _load_srcnn(weights_path, device)

    hr = lr_img.resize((lr_img.width * scale, lr_img.height * scale), Image.BICUBIC)
    hr_np = np.array(hr).astype(np.float32)
    ycbcr = _convert_rgb_to_ycbcr(hr_np)

    y = ycbcr[..., 0] / 255.0
    y = torch.from_numpy(y).to(device).unsqueeze(0).unsqueeze(0)

    with torch.no_grad():
        preds = model(y).clamp(0.0, 1.0)

    preds = preds.mul(255.0).cpu().numpy().squeeze(0).squeeze(0)
    output = np.array([preds, ycbcr[..., 1], ycbcr[..., 2]]).transpose([1, 2, 0])
    output = np.clip(_convert_ycbcr_to_rgb(output), 0.0, 255.0).astype(np.uint8)
    return Image.fromarray(output)


def _build_parser():
    p = argparse.ArgumentParser(description="No-training single-image SR (bicubic + sharpening + IBP)")
    p.add_argument("--input", required=True, help="Input image path")
    p.add_argument("--output", required=True, help="Output image path")
    p.add_argument("--scale", type=int, default=2, help="Upscale factor (>=2)")
    p.add_argument("--ibp-iters", type=int, default=10, help="IBP iterations (0 to disable)")
    p.add_argument("--ibp-lambda", type=float, default=1.0, help="IBP step size")
    p.add_argument("--sharpen-amount", type=float, default=0.9, help="Edge sharpening amount (0 to disable)")
    p.add_argument("--sharpen-radius", type=float, default=1.3, help="Edge sharpening radius")
    p.add_argument("--detail-boost", type=float, default=0.25, help="Local contrast boost (0 to disable)")
    return p


def main():
    args = _build_parser().parse_args()
    lr = Image.open(args.input).convert("RGB")
    hr = super_resolve(
        lr,
        scale=args.scale,
        sharpen_amount=args.sharpen_amount,
        sharpen_radius=args.sharpen_radius,
        detail_boost=args.detail_boost,
        ibp_iters=args.ibp_iters,
        ibp_lambda=args.ibp_lambda,
    )
    hr.save(args.output)


if __name__ == "__main__":
    main()
