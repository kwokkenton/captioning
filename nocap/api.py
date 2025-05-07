from nocap.models import clip_model_dict, ImageCaptioner, model_config
from nocap.utils import get_device, get_wandb_checkpoint_path
import torch
from nocap.dataset import Flickr30k

class ImageCaptionerAPI:
    def __init__(self, model_dict, model_config, wandb_checkpoint, device):
        checkpoint_path = get_wandb_checkpoint_path(
            wandb_checkpoint,
        )

        model = ImageCaptioner(model_dict, model_config)
        # Load the model
        checkpoint = torch.load(
            checkpoint_path,
            map_location=device, 
            weights_only=True,
        )
        model.load_state_dict(checkpoint['model_state_dict'])
        return

    def run_inference(self, image):
        caption = None

        processed_images, processed_text, attention_mask = self.model.process_batch(
                x, y,
            )
        generated = self.model.forward_sequential(processed_images[0:1])

        return caption

captioner = ImageCaptionerAPI(clip_model_dict, 
                model_config, 
                'kwokkenton-individual/mlx-week4-image-captioning/transformer_captioner:v31')

if __name__ == "__main__":
    val_ds = Flickr30k(split='val')
