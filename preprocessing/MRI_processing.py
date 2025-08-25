from torchvision.transforms import Resize as torchResize
import nibabel as nib
import torch
from monai.transforms import Transform


class GetMiddleSlice(Transform):
    def __init__(self, axis="Coronal", resize_size=[128, 128]):
        """
        Initialize the GetMiddleSlice transform.
        Args:
            axis (str): The axis along which to take the middle slice.
            resize_size (list): Target size for the output slice [height, width]
        """
        self.axis = 0 if axis == "Sagittal" else 1 if axis == "Coronal" else 2 if axis == "Axial" else None
        self.resize_size = resize_size
        super(GetMiddleSlice, self).__init__()

    def __call__(self, volume):
        # Convert NIfTI to numpy array if needed
        if isinstance(volume, str):
            volume = nib.load(volume).get_fdata()
        if isinstance(volume, nib.Nifti1Image):
            volume = volume.get_fdata()

        # Remove batch dimension if present
        volume = volume[0] if len(volume.shape) == 4 else volume

        # Get the middle slice index
        depth = volume.shape[self.axis]
        middle_idx = depth // 2

        # Helper to get a slice at a given index along the axis
        def get_slice(idx):
            if self.axis == 0:
                s = volume[idx, :, :]
            elif self.axis == 1:
                s = volume[:, idx, :]
            else:  # axis == 2
                s = volume[:, :, idx]
            return s.T  # Transpose to get correct orientation

        # Get indices for 3-channel (middle, before, after)
        idxs = [max(0, middle_idx - 1), middle_idx,
                min(depth - 1, middle_idx + 1)]
        slices = [get_slice(i) for i in idxs]

        # Convert to tensor and stack as channels
        slices_tensor = [torch.tensor(s, dtype=torch.float32) for s in slices]
        slice_tensor = torch.stack(slices_tensor, dim=0)  # [C, H, W]

        # Resize if specified
        if self.resize_size is not None:
            slice_tensor = torchResize(size=self.resize_size)(slice_tensor)

        return slice_tensor
