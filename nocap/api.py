from nocap.models import ImageCaptioner
from nocap.utils import get_wandb_checkpoint_path
import torch


class ImageCaptionerAPI:
    def __init__(self, model_dict, model_config, wandb_checkpoint, device=None):
        checkpoint_path = get_wandb_checkpoint_path(
            wandb_checkpoint,
        )
        if device is None:
            device = torch.device('cpu')

        model = ImageCaptioner(model_dict, model_config)
        # Load the model
        checkpoint = torch.load(
            checkpoint_path,
            map_location=device,
            weights_only=True,
        )
        model.load_state_dict(checkpoint['model_state_dict'], strict=False)
        model.eval()
        self.model = model
        self.show_attention = model_config.get('show_attention', False)

    def run_inference(self, image):
        """ Image is torch.Tensor or PIL image
        """
        caption = None
        # captioner.model.image_processor(val_ds[0][0], return_tensors='pt')
        image_inputs = self.model.image_processor(
            images=image,
            return_tensors='pt',
        )
        x = image_inputs['pixel_values']

        # Greedy search
        # TODO: implement beam_search
        with torch.no_grad():
            generated = torch.tensor(
                [[self.model.bos_id]], dtype=torch.int32, device=x.device,
            )
            for _ in range(self.model.text_seq_len_max - generated.size(1)):
                # assume outputs.logits [1, T, V]
                if self.show_attention: 
                    test_scores, attention = self.model.forward(x, generated)
                else: 
                    test_scores = self.model.forward(x, generated)
                # 3) Greedy pick at last position
                next_token = torch.argmax(
                    test_scores[:, -1, :], dim=-1, keepdim=True,
                )  # [1,1]
                # 4) Append and check EOS
                generated = torch.cat([generated, next_token], dim=1)  # [1, T+1]
                if next_token.item() == self.model.eos_id:
                    break

        caption = self.model.text_tokenizer.decode(generated[0][1:-1])
        if self.show_attention:
            return caption, attention, [self.model.text_tokenizer.decode(i) for i in generated[0]]
        else:
            return caption

    def __call__(self, image):
        return self.run_inference(image)
