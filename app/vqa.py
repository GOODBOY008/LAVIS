"""
 # Copyright (c) 2022, salesforce.com, inc.
 # All rights reserved.
 # SPDX-License-Identifier: BSD-3-Clause
 # For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""

import streamlit as st
import torch

from app import load_demo_image, device, blip2_model, blip2_processor
from PIL import Image


def app():
    model_type = st.sidebar.selectbox("Model:", ["BLIP"])

    # ===== layout =====
    st.markdown(
        "<h1 style='text-align: center;'>Visual Question Answering</h1>",
        unsafe_allow_html=True,
    )

    instructions = """Try the provided image or upload your own:"""
    file = st.file_uploader(instructions)

    col1, col2 = st.columns(2)

    col1.header("Image")
    if file:
        raw_img = Image.open(file).convert("RGB")
    else:
        raw_img = load_demo_image(
            "https://b2bfiles1.gigab2b.cn/image/wkseller/20822/20230216_d4f40cbadf5788d154e6fdb7e3efaf77.jpg?x-oss-process=image%2Fresize%2Cw_500%2Ch_500%2Cm_pad")

    w, h = raw_img.size
    scaling_factor = 720 / w
    resized_image = raw_img.resize((int(w * scaling_factor), int(h * scaling_factor)))

    col1.image(resized_image, use_column_width=True)
    col2.header("Question")

    user_question = col2.text_input("Input your question!", "What are objects there?")
    qa_button = st.button("Submit")

    col2.header("Answer")

    if qa_button:
        inputs = blip2_processor(images=raw_img, text="Question: " + user_question + " Answer:",
                                 return_tensors="pt").to(device, torch.float16)
        generated_ids = blip2_model.generate(**inputs)
        generated_text0 = blip2_processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
        col2.write(generated_text0 + "\n", use_column_width=True)
