from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from PIL import Image
import io
from nocap.api import ImageCaptionerAPI
from nocap.models import clip_model_dict, model_config
import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logging.basicConfig(level=logging.INFO)

captioner = ImageCaptionerAPI(
    clip_model_dict,
    model_config,
    'kwokkenton-individual/mlx-week4-image-captioning/transformer_captioner:v31',
)

app = FastAPI()

# Allow frontend to connect
app.add_middleware(
    CORSMiddleware,
    allow_origins=['*'],  # Change to your frontend's origin in production
    allow_credentials=True,
    allow_methods=['*'],
    allow_headers=['*'],
)


@app.post('/caption')
async def caption_image(file: UploadFile = File(...)):
    print('Image received')

    logger.info('Image received')
    # Read the image
    image_data = await file.read()
    image = Image.open(io.BytesIO(image_data))

    caption = captioner(image)
    logger.info(f'Caption: {caption}')

    return JSONResponse(content={'caption': caption})
