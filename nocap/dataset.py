from torch.utils.data import Dataset
from datasets import load_dataset
from torchvision import transforms
from PIL import  ImageFile
# Allow loading of truncated images
# https://github.com/python-pillow/Pillow/issues/3185
ImageFile.LOAD_TRUNCATED_IMAGES = True

class Flickr30k(Dataset):
    def __init__(self, split, image_transforms=None):
        super().__init__()

        self.ds = load_dataset('nlphuji/flickr30k')
        if split not in ['test', 'train', 'val']:
            raise ValueError(
                "fsplit {split} must be in ['test', 'train', 'val'] ",
            )
        self.split = split
        self.split_idxs = self._make_split_idxs(split)

        # Image transforms
        self.image_transforms = transforms.Compose([transforms.PILToTensor()])

    def __getitem__(self, index):
        """
        Image (torch.Tensor): (3, H, W)[torch.uint8]
        Caption (str)
        """
        # Get mapping from range to actual split indices
        split_idx = self.split_idxs[index]
        # Returns first caption, there's typically 5 captions
        return self.image_transforms(self.ds['test'][split_idx]['image']), self.ds['test'][split_idx]['caption'][0]

    def __len__(self):
        return len(self.split_idxs)

    def _make_split_idxs(self, split):
        split_idxs = []
        for idx, s in enumerate(self.ds['test']['split']):
            if s == split:
                split_idxs.append(idx)
        return split_idxs

    def __repr__(self):
        return str(f'Flickr30k {self.split} split\nNum_instances: {self.__len__()}')
