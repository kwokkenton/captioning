from nocap.models import clip_model_dict, model_config
from nocap.api import ImageCaptionerAPI
from nocap.utils import get_device, get_wandb_checkpoint_path
import torch
from nocap.dataset import Flickr30k
import matplotlib.pyplot as plt
import random

captioner = ImageCaptionerAPI(clip_model_dict, 
                model_config, 
                'kwokkenton-individual/mlx-week4-image-captioning/transformer_captioner:v31',
                device = torch.device('cpu'))


val_ds = Flickr30k(split='val')
choice = random.randint(0, len(val_ds) -1 )
im, target = val_ds[choice]
plt.imshow(im.permute(1,2,0))
plt.title(f'GT: {target} \nPrediction: {captioner.run_inference(im)}')
plt.show()