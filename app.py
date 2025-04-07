import os
os.environ["HF_HUB_ENABLE_HF_TRANSFER"]='1'
from huggingface_hub import snapshot_download
import re
from transformers import DonutProcessor, VisionEncoderDecoderModel
from PIL import Image
import requests
import torch

class InferlessPythonModel:
    def initialize(self):
        model_id = "naver-clova-ix/donut-base-finetuned-docvqa"
        snapshot_download(repo_id=model_id,allow_patterns=["*.bin"])
        self.processor = DonutProcessor.from_pretrained(model_id)
        self.model = VisionEncoderDecoderModel.from_pretrained(model_id).to("cuda")
        
    def infer(self,inputs):
        user_question = inputs["user_question"]
        image_url = inputs["image_url"]
        image = Image.open(requests.get(image_url, stream=True).raw)
        prompt = f"<s_docvqa><s_question>{user_question}</s_question><s_answer>"

        decoder_input_ids = self.processor.tokenizer(prompt, add_special_tokens=False, return_tensors="pt").input_ids
        pixel_values = self.processor(image, return_tensors="pt").pixel_values

        outputs = self.model.generate(
        pixel_values.to("cuda"),
        decoder_input_ids=decoder_input_ids.to("cuda"),
        max_length=self.model.decoder.config.max_position_embeddings,
        pad_token_id=self.processor.tokenizer.pad_token_id,
        eos_token_id=self.processor.tokenizer.eos_token_id,
        use_cache=True,
        bad_words_ids=[[self.processor.tokenizer.unk_token_id]],
        return_dict_in_generate=True,)
    
        sequence = self.processor.batch_decode(outputs.sequences)[0]
        sequence = sequence.replace(self.processor.tokenizer.eos_token, "").replace(self.processor.tokenizer.pad_token, "")
        sequence = re.sub(r"<.*?>", "", sequence, count=1).strip()  # remove first task start token
        return self.processor.token2json(sequence)

    def finalize(self):
        pass
