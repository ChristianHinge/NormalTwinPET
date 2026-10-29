import argparse

import nibabel as nib
import torch
import torchio as tio

from .net import UNet
from .stitcher import GennormStitcher
from .transforms import InPlaneCropOrPad
from .weights import get_weights


def predict(ct_path, device):
    unet = UNet(n_conv_layers=4, n_channels_in=1)
    unet.load_state_dict(torch.load(get_weights(), map_location="cpu", weights_only=True))
    unet.to(device).eval()

    stitcher = GennormStitcher(
        edge_to_center_ratio=0.3,
        patch_overlap=[0, 0, 0],
        batch_size=2,
        beta=3,
        patch_size=(192, 192, 32),
        device=device,
    )

    original = tio.ScalarImage(ct_path)
    transform = tio.Compose([
        tio.ToCanonical(),
        tio.Resample(target=(2, 2, 2)),
        InPlaneCropOrPad((256, 256)),
        tio.Clamp(-500, 600),
        tio.RescaleIntensity(out_min_max=(0, 2)),
    ])
    ct = transform(original)

    ntpet = stitcher(unet, {"ct": ct})
    ntpet = tio.ScalarImage(tensor=ntpet, affine=ct.affine)
    # Back onto the input CT grid so the output overlays the CT voxel-for-voxel
    return tio.Resample(target=original)(ntpet)


def main():
    parser = argparse.ArgumentParser(
        description="Generate a normal twin PET (image-derived lean-body-mass SUV) from a CT image."
    )
    parser.add_argument("-i", "--input", required=True, help="Input CT (NIfTI)")
    parser.add_argument("-o", "--output", required=True, help="Output normal twin PET (NIfTI)")
    parser.add_argument(
        "-d", "--device", default="cuda" if torch.cuda.is_available() else "cpu",
        help="Torch device (default: cuda if available, else cpu)",
    )
    args = parser.parse_args()

    ntpet = predict(args.input, torch.device(args.device))
    arr = ntpet.numpy().squeeze().astype("float32")
    nib.save(nib.Nifti1Image(arr, ntpet.affine), args.output)


if __name__ == "__main__":
    main()
