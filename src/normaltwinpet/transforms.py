import torchio as tio

class InPlaneCropOrPad(tio.SpatialTransform):
    def __init__(self, size, fill_value=-1024, **kwargs):
        super().__init__(**kwargs)
        self.size = size                  # (x, y), or one int for square
        self.fill_value = fill_value

    def apply_transform(self, subject):
        _, _, d = subject.spatial_shape   # keep current number of slices
        x, y = (self.size, self.size) if isinstance(self.size, int) else self.size
        return tio.CropOrPad((x, y, d),padding_mode=self.fill_value)(subject)