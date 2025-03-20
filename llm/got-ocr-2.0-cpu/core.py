from transformers import AutoModel, AutoTokenizer
from typing import Optional, Literal, Union
from pathlib import Path
from io import BytesIO
from PIL import Image as PILImage


class GOTOCRProcessor:
    def __init__(self):
        self._model_cache = {}

    def get_model_and_tokenizer(self):
        cache_key = "got_ocr_model"
        if cache_key not in self._model_cache:
            tokenizer = AutoTokenizer.from_pretrained(
                "srimanth-d/GOT_CPU", trust_remote_code=True
            )
            model = AutoModel.from_pretrained(
                "srimanth-d/GOT_CPU",
                trust_remote_code=True,
                low_cpu_mem_usage=True,
                use_safetensors=True,
                pad_token_id=tokenizer.eos_token_id,
            )
            self._model_cache[cache_key] = {
                "model": model.eval(),
                "tokenizer": tokenizer,
            }
        return self._model_cache[cache_key]["model"], self._model_cache[cache_key][
            "tokenizer"
        ]

    def process_image(
        self,
        image: Union[str, Path, BytesIO],
        *,
        ocr_type: Literal["ocr", "format"] = "ocr",
        method: Literal["chat", "chat_crop"] = "chat",
        render: bool = False,
        save_render_file: Optional[str | Path] = None,
        ocr_box: Optional[str] = None,
        ocr_color: Optional[str] = None,
    ) -> str:
        model, tokenizer = self.get_model_and_tokenizer()

        processor = model.chat_crop if method == "chat_crop" else model.chat

        kwargs = {
            "ocr_type": ocr_type,
            "render": render,
            "save_render_file": save_render_file,
            "gradio_input": True,
        }

        if ocr_box:
            kwargs["ocr_box"] = ocr_box
        if ocr_color:
            kwargs["ocr_color"] = ocr_color

        im = PILImage.open(image).convert('RGB')
        return processor(tokenizer, im, **kwargs)

    def clear_cache(self):
        self._model_cache.clear()
