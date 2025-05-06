# Image Captioning with Transformers

Task
- Generate image captions

shared latent space of both vision and text

Use `transformers` library
- Wrappers around pre-trained models

### CLIP

CLIP jointly represents images and text decriptions.
- 2 encoders
    - Vision encoder: ViT
    - Text encoder: A 63M-parameter 12layer 512-wide model with 8 attention heads
- Implementation on HuggingFace
    -   CLIPTextModelWithProjection
    -   Outputs a sentence projection, which are pooled (EOS token) states
    - Automatically pads with [SOS] and [EOS] tokens. The pad token is also the [EOS] token

Tokeniser should belong in the collate_fn to take advantage of the parallelisable inputs.


## References

Mokady, Ron, Amir Hertz, and Amit H. Bermano. ‘ClipCap: CLIP Prefix for Image Captioning’. arXiv, 18 November 2021. https://doi.org/10.48550/arXiv.2111.09734.
