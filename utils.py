import numpy as np
import torch


def _image_to_sequence(image, batch_size=1, image_size=64, patch_size=4, in_channels=3, stride=2, **kwargs):

    # if in_channels != 3:
    #     return
    width_num = height_num = (image_size - patch_size) // stride + 1
    sequence_len = width_num * height_num
    patch_area = patch_size ** 2
    patch_len = in_channels * patch_area + 2
    diff = image_size - patch_size + 1

    # complexity: O(patch_size^2)
    sequence = torch.full([batch_size, sequence_len, patch_len], 0)
    for i in range(patch_size):
        for j in range(patch_size):
            sequence[:, :, i * patch_size + j] = image[:, 0, i:i + diff:stride, j:j + diff:stride].reshape(batch_size, -1)
            sequence[:, :, i * patch_size + j + patch_area] = image[:, 1, i:i + diff:stride, j:j + diff:stride].reshape(batch_size, -1)
            sequence[:, :, i * patch_size + j + 2*patch_area] = image[:, 2, i:i + diff:stride, j:j + diff:stride].reshape(batch_size, -1)

    # add 2 dimension for coordinates recording
    coord = np.arange(width_num)
    rol_coord = coord.repeat(height_num)
    col_coord = np.tile(coord, height_num)
    sequence[:, :, -2] = torch.from_numpy(rol_coord)
    sequence[:, :, -1] = torch.from_numpy(col_coord)

    return sequence

class ImageToSequence():

    def __init__(self, batch_size=1, image_size=64, patch_size=4, in_channels=3, stride=2):
        # if image.size() != [3, 64, 64]:
        #     return
        self.kwargs = dict(batch_size=batch_size, image_size=image_size, patch_size=patch_size, in_channels=in_channels, stride=stride)

    def __call__(self, image):
        return _image_to_sequence(image, **self.kwargs)