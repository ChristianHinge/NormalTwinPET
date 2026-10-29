# normaltwinpet

Generate a whole-body **normal twin PET** from a CT image, using the
CT-only model from:

> Hinge et al., *Normal twin PET: personalized generative modeling for confounder
> correction and anomaly detection in whole-body PET/CT*, Scientific Reports 15,
> 42662 (2025). https://doi.org/10.1038/s41598-025-26827-y

Input is a CT, output is the normal twin PET, on the same voxel grid as the CT.

The output is normalized as image-derived lean-body-mass SUV (see
[ID-uptake-values](https://github.com/christianhinge/ID-uptake-values)).

## Usage

```bash
pip install normaltwinpet
normaltwinpet -i ct.nii.gz -o ntpet.nii.gz
```

Uses CUDA if available, otherwise CPU (slow). Override with `--device`.

On first run the weights (~250 MB) are downloaded from the GitHub release and
cached in the user cache directory (`~/.cache/normaltwinpet` on Linux). Set
`NORMALTWINPET_HOME` to use another location.

## Notes

The CT is resampled to 2 mm and cropped/padded to 256x256 in-plane for the
network, then resampled back to the input grid. Anatomy outside the central
256x256 (512 mm) in-plane field of view is not predicted and will be zero.
