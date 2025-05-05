from torch import nn
import torch

def make_causal_mask(prefix_len, sequence_len, device):
    total_length = prefix_len + sequence_len
    square_causal_mask = torch.triu(torch.ones(total_length, total_length, device=device), diagonal=1)
    square_causal_mask[0:prefix_len, 0:prefix_len] = 0
    # mask = torch.cat([torch.ones((prefix_len, prefix_len), device=device), square_causal_mask], dim=1)
    return square_causal_mask

class ImageCaptioner(nn.Module):
    def __init__(self, model_dict:dict):
        super().__init__()
        self.vision_model = model_dict.get('vision_model')
        self.text_model = model_dict.get('text_model')
        # The processors can do parellelisable inputs
        self.image_processor = model_dict.get('image_processor')
        self.text_tokenizer = model_dict.get('text_tokeniser')
        self.vision_hidden_dim = model_dict.get('vision_hidden_dim')
        self.text_hidden_dim = model_dict.get('text_hidden_dim')
        self.vocab_size = len(self.text_tokenizer)
        
        # These are chosen
        self.hidden_dim = 512
        self.heads = 8
        self.num_layers = 6

        # These are linked to CLIP
        self.seq_len_max = 50 + 76
        
        # Layers
        self.vision_proj = nn.Linear(self.vision_hidden_dim, self.hidden_dim)
        self.text_proj = nn.Linear(self.text_hidden_dim, self.hidden_dim)
        # We use nn.TransformerEncoder as nn.TransformerDecoder expects cross
        # attention 
        # https://www.reddit.com/r/MLQuestions/comments/102o11v/is_gpt_actually_using_the_encoder_not_the_decoder/
        decoder_layer = nn.TransformerEncoderLayer(d_model=self.hidden_dim, 
                                                   nhead=self.heads, 
                                                   batch_first=True)
        self.decoder = nn.TransformerEncoder(decoder_layer, self.num_layers)
        self.language_head = nn.Linear(self.hidden_dim, self.vocab_size)
        
        # Positional encodings
        self.positional_encodings = nn.Parameter(
            torch.empty(1, self.seq_len_max, self.hidden_dim),
        )
        nn.init.normal_(self.positional_encodings, std = 0.02)

    
    def forward(self, image_inputs:dict, text_inputs:dict):
        """
        image_inputs
        TODO: reshape into a tensor

        text_inputs: reshape
        """
        # Get embeddings from the pretrained models for an easy start
        image_embeddings = self.get_image_embeddings(image_inputs)
        B, N_image_tokens, _ = image_embeddings.shape
        text_embeddings = self.get_text_embeddings(text_inputs)
        _, N_text_tokens, _ = text_embeddings.shape
        print(image_embeddings.shape, text_embeddings.shape)

        # Need a linear layer on top of both to reshape into common dimensions
        # Need to add positional encodings
        res = torch.cat([self.vision_proj(image_embeddings), 
                         self.text_proj(text_embeddings)], dim=1)
        # Causal masking for text_inputs
        causal_mask = make_causal_mask(N_image_tokens, N_text_tokens, res.device)
        res = self.decoder(res, mask=causal_mask, src_key_padding_mask = None)
        out = self.language_head(res[:,N_image_tokens:])
        return out
    
    def get_image_embeddings(self, vision_inputs):
        # X_im is the result from the processor
        # Shape is B, N_tokens = 50, D_model = 768
        # TODO figure out how to parallelise this
        return self.vision_model(**vision_inputs).last_hidden_state

    def get_text_embeddings(self, text_inputs):
        # Shape is B, N_tokens, D_model = 512
        # this is also a possibility captioning_model.text_model(text_inputs.input_ids).last_hidden_state
        return self.text_model(**text_inputs).last_hidden_state