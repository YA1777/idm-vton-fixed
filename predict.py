import cog
import os
import torch
import numpy as np
from PIL import Image
from pathlib import Path
import sys

from huggingface_hub import snapshot_download

MODEL_ID = "yisol/IDM-VTON"
CACHE_DIR = "/src/checkpoints"


class Predictor(cog.Predictor):
    def setup(self):
        self.device = torch.device("cuda")

        model_path = Path(CACHE_DIR)
        model_path.mkdir(parents=True, exist_ok=True)

        print(f"Downloading model {MODEL_ID} to {model_path}...")
        snapshot_download(
            MODEL_ID,
            local_dir=model_path,
            ignore_patterns=["*.msgpack", "*.h5"],
        )
        print("Model downloaded successfully!")

        from src.tryon_pipeline import StableDiffusionXLInpaintPipeline
        from src.unet_hacked_garmnet import UNet2DConditionModel
        from src.unet_hacked_tryon import UNet2DConditionModel as TryonUNet
        from transformers import CLIPImageProcessor
        from diffusers import AutoencoderKL

        self.vae = AutoencoderKL.from_pretrained(
            "stabilityai/sd-vae-ft-mse", torch_dtype=torch.float16
        ).to(self.device)

        self.unet = UNet2DConditionModel.from_pretrained(
            model_path, subfolder="unet", torch_dtype=torch.float16
        ).to(self.device)

        self.unet_encoder = UNet2DConditionModel.from_pretrained(
            model_path, subfolder="unet_encoder", torch_dtype=torch.float16
        ).to(self.device)

        self.pipe = StableDiffusionXLInpaintPipeline(
            vae=self.vae,
            text_encoder=None,
            text_encoder_2=None,
            tokenizer=None,
            tokenizer_2=None,
            unet=self.unet,
            unet_encoder=self.unet_encoder,
            scheduler=None,
            feature_extractor=CLIPImageProcessor(),
            requires_safety_checker=False,
        )

    def run(self, human_img: cog.Path, garm_img: cog.Path,
            category: str = "upper_body") -> cog.Path:
        person = Image.open(str(human_img)).resize((384, 512))
        cloth = Image.open(str(garm_img)).resize((384, 512))
        person_np = np.array(person)

        from src.preprocess.openpose.run_openpose import OpenPose
        from src.preprocess.humanparsing.run_parsing import Parsing
        from src.preprocess.densepose import get_densepose

        openpose = OpenPose()
        parsing = Parsing()

        pose_data = openpose(person.resize((384, 512)))
        parse_map, _ = parsing(person.resize((384, 512)))
        densepose = get_densepose(person_np)

        labels_map = {"upper_body": [5,6,7], "lower_body": [8,9], "dresses": [5,6,7,8,9]}
        labels = labels_map.get(category, [5,6,7])
        mask = np.isin(parse_map, labels).astype(np.uint8) * 255
        mask_3ch = np.stack([mask]*3, axis=-1)
        agnostic = np.where(mask_3ch > 0, np.ones_like(person_np)*128, person_np).astype(np.uint8)

        with torch.no_grad():
            result = self.pipe(
                image=Image.fromarray(agnostic),
                cloth=cloth,
                pose_img=Image.fromarray(densepose),
                num_inference_steps=30,
                guidance_scale=2.0,
                height=512,
                width=384,
            ).images[0]

        output = Path("/tmp/output.jpg")
        output.parent.mkdir(parents=True, exist_ok=True)
        result.save(str(output))
        return output
