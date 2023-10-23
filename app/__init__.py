"""
 # Copyright (c) 2022, salesforce.com, inc.
 # All rights reserved.
 # SPDX-License-Identifier: BSD-3-Clause
 # For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""

from PIL import Image
import requests

import streamlit as st
import torch
from transformers import Blip2Processor, Blip2ForConditionalGeneration

blip2_name = "Salesforce/blip2-opt-6.7b"


@st.cache_data()
def load_demo_image(img_url=None):
    if img_url is None:
        img_url = (
            "https://storage.googleapis.com/sfr-vision-language-research/BLIP/demo.jpg"
        )
    raw_image = Image.open(requests.get(img_url, stream=True).raw).convert("RGB")
    return raw_image


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_blip2_model():
    return Blip2ForConditionalGeneration.from_pretrained(
        blip2_name, torch_dtype=torch.float16
    ).to(device)


def get_blip2_processor():
    return Blip2Processor.from_pretrained(blip2_name)


cache_root = "/export/home/.cache/lavis/"

blip2_model = get_blip2_model()
blip2_processor = get_blip2_processor()
