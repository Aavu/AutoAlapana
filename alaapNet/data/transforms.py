import torch
import numpy as np


class Transform(torch.nn.Module):
    def __init__(self):
        super().__init__()


class ToTensor(Transform):
    """Convert ndarrays in sample to Tensors."""
    def __init__(self):
        super(ToTensor, self).__init__()

    def forward(self, sample: np.ndarray):
        return torch.tensor(sample, requires_grad=True, dtype=torch.float)


class FillGaps(Transform):
    def __init__(self, fill_type='interpolate'):
        super().__init__()
        self.fill_type = fill_type

    def forward(self, x):
        return self.fill_gaps(x)

    def fill_gaps(self, data: np.ndarray, threshold: int = 50, zero_threshold: float = 0.001):
        """
        Fill space between 2 notes in midi contour if the space is < 10 samples. Call this function for each frame
        :param data: The array to be filled
        :param threshold: (Optional) number of zeros below which they should be fixed
        :param zero_threshold: (Optional) The zero threshold to compare
        :return filled data
        """

        data = self.fill_na(data)
        _i = 1
        while _i < len(data):
            if data[_i] < zero_threshold:
                num_zeros = 1
                for _j in range(_i + 1, len(data)):
                    if data[_j] < zero_threshold:
                        num_zeros += 1
                    else:
                        if np.abs(data[_i - 1] - data[_j]) > 0.01 and num_zeros < threshold:
                            data[_i:_j] = data[_i - 1]
                        _i = _j
                        break
            _i += 1

        return data


    def fill_na(self, arr, value=0):
        arr[np.isnan(arr)] = value
        arr[np.isinf(arr)] = value
        return arr
