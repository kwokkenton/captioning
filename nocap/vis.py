from io import BytesIO
import base64
import torch.nn.functional as F
import torch
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np

def visualize_text_to_image_attention(attn_weights, text_tokens, image_shape=(7, 7), token_idx=0):
    """
    Visualize attention from a specific text token to image patches.
    
    Parameters:
        attn_weights: Tensor (num_heads, seq_len_q, seq_len_kv)
        text_tokens: List of str — the decoded text tokens (length 13 in your example)
        image_shape: Tuple — spatial shape of image patch grid (e.g., (7, 7))
        head: int — attention head index
        token_idx: int — index of the text token (0 to 12)

    """
    assert attn_weights.shape[1] == attn_weights.shape[2], "Expected square attention matrix"
    # assert attn_weights.shape[1] >= len(text_tokens) + np.prod(image_shape), "Invalid token sizes"

    num_heads = attn_weights.shape[0]
    num_img = np.prod(image_shape)
    text_start = num_img

    # Extract attention for a single head and text token
    # First token is the CLS token
    attn = attn_weights[:, text_start + token_idx, 1:num_img+1].mean(axis=0)  # shape (50,)
    attn = attn.detach().cpu().numpy()
    attn = attn.reshape(image_shape)
    return attn

def get_attention_maps(model, im, tokens, attn, layer = 1):
    N_tokens = len(tokens)
    encoded_images = []

    for token_idx in range(N_tokens):
        attn_mat = visualize_text_to_image_attention(attn[layer][0], tokens, token_idx=token_idx)

        # Convert to tensor and unsqueeze to shape [1, 1, H, W] for interpolation
        attn_tensor = torch.tensor(attn_mat).unsqueeze(0).unsqueeze(0)  # [1, 1, 7, 7]
        # Resize to match the image shape (e.g., 224x224)
        attn_resized = F.interpolate(attn_tensor, size=(224,224), mode='bilinear', align_corners=False)
        attn_resized = attn_resized.squeeze().numpy()  # Shape: (224, 224)


        mean = torch.tensor([0.48145466, 0.4578275, 0.40821073])
        std = torch.tensor([0.26862954, 0.26130258, 0.27577711])
        # plt.title(f'{tokens[token_idx]}')
        plt.imshow(2*(model.image_processor(images= [im], return_tensors='pt')['pixel_values'][0].permute(1,2,0) + mean) * std  )
        plt.imshow(attn_resized, alpha=0.5)
        plt.axis('off')

        buf = BytesIO()
        plt.savefig(buf, format='PNG')
        buf.seek(0)
        img = Image.open(buf)

        # Convert the image to base64 for sending it in the response
        img_byte_arr = BytesIO()
        img.save(img_byte_arr, format='PNG')
        encoded_img = base64.b64encode(img_byte_arr.getvalue()).decode('utf-8')

        encoded_images.append({
            'token': tokens[token_idx],
            'attention_image': encoded_img
        })

        # encoded_images.append(encoded_img)

    encoded_images = encoded_images[1:-1]
    return encoded_images