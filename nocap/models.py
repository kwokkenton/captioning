from torch import nn
import torch
from transformers import AutoProcessor, CLIPVisionModel, CLIPTextModel, AutoTokenizer
from copy import deepcopy

clip_model_dict = {
    'vision_model': CLIPVisionModel.from_pretrained('openai/clip-vit-base-patch32'),
    'text_model': CLIPTextModel.from_pretrained('openai/clip-vit-base-patch32').text_model.embeddings,
    'image_processor': AutoProcessor.from_pretrained('openai/clip-vit-base-patch32', use_fast=True),
    'text_tokeniser': AutoTokenizer.from_pretrained('openai/clip-vit-base-patch32'),
    'vision_hidden_dim': 768,
    'text_hidden_dim': 512,
}


def get_text_inputs_and_targets(tokenised_text: torch.Tensor, attention_mask, target_id: int, replacement_id: int) -> tuple[dict, torch.Tensor]:
    """ processed_text is the CLIP-tokenised text
    This function produces the inputs to the model and the targets for evaluation
    on the auto-regressive task.

    """

    # Want the model to predict all but the last token and so we slice until it
    tokenised_text = deepcopy(tokenised_text)
    text_inputs = tokenised_text[:, :-1]
    attention_mask = deepcopy(attention_mask)[:, :-1]

    # Targets are all but the first token (don't predict <bos>)
    text_targets = process_padding(
        tokenised_text[:, 1:], target_id, replacement_id,
    )
    return text_inputs, attention_mask, text_targets


def make_causal_mask(prefix_len: int, sequence_len: int, device: torch.device):
    """ Image tokens (of prefix_len) are in the prefix, which can self attend.
        Text tokens (of sequence_len) cannot attend to future tokens
    """
    total_length = prefix_len + sequence_len
    square_causal_mask = torch.triu(
        torch.ones(
            total_length, total_length, device=device, dtype=torch.bool,
        ), diagonal=1,
    )
    square_causal_mask[0:prefix_len, 0:prefix_len] = 0
    # mask = torch.cat([torch.ones((prefix_len, prefix_len), device=device), square_causal_mask], dim=1)
    return square_causal_mask


def make_src_key_padding_mask(prefix_len, attention_mask, device: torch.device):
    """
    a binary mask of shape (N,S) indicating which elements within key to ignore for
    the purpose of attention (i.e. treat as “padding”). For unbatched query,
    shape should be (S). Binary and float masks are supported. For a binary
    mask, a True value indicates that the corresponding key value will be
    ignored for the purpose of attention.

    https://docs.pytorch.org/docs/stable/generated/torch.nn.MultiheadAttention.html
    """
    B, sequence_len = attention_mask.shape
    attention_mask = attention_mask.to(dtype=torch.bool)
    zeros_to_prepend = torch.zeros(
        (B, prefix_len), dtype=torch.bool, device=device,
    )
    result = torch.cat((zeros_to_prepend, attention_mask), dim=1)
    return result


def process_padding(x: torch.Tensor, target: int, replacement: int) -> torch.Tensor:
    """
    Problem: Padding for the CLIP model is same as the <eos> token.
    We replace all <eos> tokens (target) but the first one for each entry with
        the replacement token, typically the <bos> token for simplicity.

    We then set the loss function to ignore the <bos> token separately.
    """
    # target = 49407
    # replacement = 49406 # also the bos token

    # Create a copy to modify
    x_modified = x.clone()

    for i in range(x.size(0)):  # loop over rows
        row = x[i]
        mask = row == target
        indices = torch.nonzero(mask, as_tuple=True)[0]
        if len(indices) > 1:
            # Replace all but the first occurrence
            x_modified[i, indices[1:]] = replacement
    return x_modified


def collate_fn(batch: list[tuple[torch.Tensor, str]]) -> tuple[list[torch.Tensor], list[dict]]:
    xs = []
    ys = []
    for item in batch:
        xs.append(item[0])
        ys.append(item[1])
    return xs, ys


class ImageCaptioner(nn.Module):
    def __init__(self, model_dict: dict, model_config: dict):
        super().__init__()
        self.vision_model = model_dict.get('vision_model')
        self.text_model = model_dict.get('text_model')

        self.vision_model.eval()
        self.text_model.eval()

        for param in self.vision_model.parameters():
            param.requires_grad = False
        for param in self.text_model.parameters():
            param.requires_grad = False

        # The processors can do parellelisable inputs
        self.image_processor = model_dict.get('image_processor')
        self.text_tokenizer = model_dict.get('text_tokeniser')
        self.vision_hidden_dim = model_dict.get('vision_hidden_dim')
        self.text_hidden_dim = model_dict.get('text_hidden_dim')
        self.vocab_size = len(self.text_tokenizer)

        # These are chosen
        self.hidden_dim = model_config.get('hidden_dim')  # 512
        self.heads = model_config.get('num_heads')  # 8
        self.num_layers = model_config.get('num_layers')  # 6

        # These are linked to CLIP
        self.text_seq_len_max = 77
        self.seq_len_max = 50 + self.text_seq_len_max

        self.bos_id, self.eos_id = self.text_tokenizer([''])['input_ids'][0]
        self.pad_id = self.bos_id
        # Layers
        self.vision_proj = nn.Linear(self.vision_hidden_dim, self.hidden_dim)
        self.text_proj = nn.Linear(self.text_hidden_dim, self.hidden_dim)
        # We use nn.TransformerEncoder as nn.TransformerDecoder expects cross
        # attention
        # https://www.reddit.com/r/MLQuestions/comments/102o11v/is_gpt_actually_using_the_encoder_not_the_decoder/
        decoder_layer = nn.TransformerEncoderLayer(
            d_model=self.hidden_dim,
            nhead=self.heads,
            batch_first=True,
        )
        self.decoder = nn.TransformerEncoder(decoder_layer, self.num_layers)
        self.language_head = nn.Linear(self.hidden_dim, self.vocab_size)

        # Positional encodings
        self.positional_encodings = nn.Parameter(
            torch.empty(1, self.seq_len_max, self.hidden_dim),
        )
        nn.init.normal_(self.positional_encodings, std=0.02)

    def forward(self, image_inputs: torch.Tensor, text_inputs: torch.Tensor, text_attention_mask=None) -> torch.Tensor:
        """
        image_inputs (dict): key 'pixel_values' torch.Size([B, 3, 224, 224])

        text_inputs: reshape
        """
        # Get embeddings from the pretrained models for an easy start
        image_embeddings = self.get_image_embeddings(image_inputs)
        B, N_image_tokens, _ = image_embeddings.shape
        text_embeddings = self.get_text_embeddings(text_inputs)
        _, N_text_tokens, _ = text_embeddings.shape

        # Need a linear layer on top of both to reshape into common dimensions
        res = torch.cat(
            [
                self.vision_proj(image_embeddings),
                self.text_proj(text_embeddings),
            ], dim=1,
        )
        # Add positional embeddings
        res = res + self.positional_encodings[:, :res.shape[1], :]
        # Causal masking for text_inputs
        causal_mask = make_causal_mask(
            N_image_tokens, N_text_tokens, res.device,
        )
        # If an attention mask is provided (during training)
        if text_attention_mask is None:
            src_key_padding_mask = None
        else:
            src_key_padding_mask = make_src_key_padding_mask(
                N_image_tokens,
                1 - text_attention_mask,
                text_attention_mask.device,
            )
        res = self.decoder(
            res, mask=causal_mask,
            src_key_padding_mask=src_key_padding_mask,
        )
        # Return sequence tokens output and ignore the prefix
        out = self.language_head(res[:, N_image_tokens:])
        return out

    def process_batch(self, image_inputs: list, text_inputs: list) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # Returns dicts
        image_inputs = self.image_processor(
            images=image_inputs,
            return_tensors='pt',
        )
        text_inputs = self.text_tokenizer(
            text_inputs,
            padding=True,
            return_tensors='pt',
        )
        # Clip if longer than the longest sequence length possible
        if text_inputs.input_ids.shape[1] > self.text_seq_len_max:
            input_ids = text_inputs['input_ids'][
                :,
                :self.text_seq_len_max,
            ].clone()
            attention_mask = text_inputs['attention_mask'][
                :,
                :self.text_seq_len_max,
            ].clone()
        else:
            input_ids = text_inputs['input_ids']
            attention_mask = text_inputs['attention_mask']

        return image_inputs['pixel_values'], input_ids, attention_mask

    def get_image_embeddings(self, vision_inputs: dict):
        # X_im is the result from the processor
        # Shape is B, N_tokens = 50, D_model = 768
        if type(vision_inputs) is torch.Tensor:
            return self.vision_model(vision_inputs).last_hidden_state
        else:
            return self.vision_model(**vision_inputs).last_hidden_state

    def get_text_embeddings(self, text_inputs):
        # Shape is B, N_tokens, D_model = 512
        # this is also a possibility captioning_model.text_model(text_inputs.input_ids).last_hidden_state
        if type(text_inputs) is not torch.Tensor:
            return self.text_model.token_embedding(text_inputs['input_ids'])
        else:
            return self.text_model.token_embedding(text_inputs)

    def trainable_params(self):
        return filter(lambda p: p.requires_grad, self.parameters())

    def forward_sequential(self, x: torch.Tensor):
        # assert type(x) is torch.Tensor
        generated = torch.tensor(
            [[self.bos_id]], dtype=torch.int32, device=x.device,
        )
        for _ in range(self.text_seq_len_max - generated.size(1)):
            # assume outputs.logits [1, T, V]
            test_scores = self.forward(x, generated)
            # 3) Greedy pick at last position
            next_token = torch.argmax(
                test_scores[:, -1, :], dim=-1, keepdim=True,
            )  # [1,1]
            # 4) Append and check EOS
            generated = torch.cat([generated, next_token], dim=1)  # [1, T+1]
            if next_token.item() == self.eos_id:
                break
        return generated


def calculate_accuracy(scores, y, pad_token_id=None):
    # scores: B, N, N_classes
    # y: B, N
    if pad_token_id is not None:
        # Remove padding tokens from the target
        mask = y != pad_token_id
        scores = scores[mask]
        y = y[mask]
    correct = torch.sum(scores.argmax(dim=-1) == y)
    return correct/y.numel(), correct
