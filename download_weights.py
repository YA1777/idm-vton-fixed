#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""使用 requests 下载 HuggingFace 文件"""
import os
import requests
from pathlib import Path

BASE_URL = "https://hf-mirror.com/yisol/IDM-VTON/resolve/main"
LOCAL_DIR = Path("D:/虚拟试衣/replicate-model/src/checkpoints")

# 要下载的文件列表
FILES = [
    ".gitattributes",
    "README.md",
    "assets/teaser.png",
    "assets/teaser2.png",
    "densepose/model_final_162be9.pkl",
    "humanparsing/parsing_atr.onnx",
    "humanparsing/parsing_lip.onnx",
    "image_encoder/config.json",
    "image_encoder/model.safetensors",
    "model_index.json",
    "openpose/ckpts/body_pose_model.pth",
    "scheduler/scheduler_config.json",
    "text_encoder/config.json",
    "text_encoder/model.safetensors",
    "text_encoder_2/config.json",
    "text_encoder_2/model.safetensors",
    "tokenizer/merges.txt",
    "tokenizer/special_tokens_map.json",
    "tokenizer/tokenizer_config.json",
    "tokenizer/vocab.json",
    "tokenizer_2/merges.txt",
    "tokenizer_2/special_tokens_map.json",
    "tokenizer_2/tokenizer_config.json",
    "tokenizer_2/vocab.json",
    "unet/config.json",
    "unet/diffusion_pytorch_model.bin",
    "unet_encoder/config.json",
    "unet_encoder/diffusion_pytorch_model.safetensors",
    "vae/config.json",
    "vae/diffusion_pytorch_model.safetensors",
]

def download_file(relative_path):
    """下载单个文件"""
    url = f"{BASE_URL}/{relative_path}"
    local_path = LOCAL_DIR / relative_path

    # 创建目录
    local_path.parent.mkdir(parents=True, exist_ok=True)

    if local_path.exists() and local_path.stat().st_size > 0:
        print(f"Skip (exists): {relative_path}")
        return True

    print(f"Downloading: {relative_path}")
    try:
        response = requests.get(url, stream=True, timeout=120)
        response.raise_for_status()

        total_size = int(response.headers.get('content-length', 0))
        downloaded = 0

        with open(local_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
                    downloaded += len(chunk)
                    if total_size > 0:
                        pct = downloaded * 100 // total_size
                        print(f"\r  {pct}%", end='', flush=True)

        print()
        print(f"  Saved: {local_path.name} ({local_path.stat().st_size / 1024 / 1024:.1f} MB)")
        return True
    except Exception as e:
        print(f"  Error: {e}")
        return False

def main():
    success = 0
    failed = []

    for filepath in FILES:
        if download_file(filepath):
            success += 1
        else:
            failed.append(filepath)

    print(f"\nDone! {success}/{len(FILES)} files downloaded")
    if failed:
        print(f"Failed: {failed}")

if __name__ == "__main__":
    main()
