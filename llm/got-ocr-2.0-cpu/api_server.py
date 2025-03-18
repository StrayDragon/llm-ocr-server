from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.concurrency import asynccontextmanager
from fastapi.responses import HTMLResponse
import os
import uuid
import base64
from typing import Optional
from enum import Enum
from transformers import AutoModel, AutoTokenizer
import shutil
import tempfile
from pydantic import BaseModel, Field


MODEL_CACHE = {}


def get_model_and_tokenizer():
    cache_key = "got_ocr_model"
    if cache_key not in MODEL_CACHE:
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
        MODEL_CACHE[cache_key] = {"model": model.eval(), "tokenizer": tokenizer}
    return MODEL_CACHE[cache_key]["model"], MODEL_CACHE[cache_key]["tokenizer"]


@asynccontextmanager
async def lifespan(_: FastAPI):
    get_model_and_tokenizer()
    yield
    MODEL_CACHE.clear()


app = FastAPI(
    title="GOT OCR 2.0 API",
    description="API for extracting text from images using GOT OCR 2.0 model.",
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/")
def handle_all():
    return {"message": "GOT OCR 2.0 API Server"}


class OcrType(str, Enum):
    OCR = "ocr"
    FORMAT = "format"


class Method(str, Enum):
    CHAT = "chat"
    CHAT_CROP = "chat_crop"


class OCRResponse(BaseModel):
    result: str = Field(..., description="Extracted text from the image")


class ErrorResponse(BaseModel):
    detail: str = Field(..., description="Error message")


@app.post("/ocr", response_model=OCRResponse)
async def process_image(
    by_file: Optional[UploadFile] = File(None, description="Image file to process"),
    by_base64: Optional[str] = Form(None, description="Base64 encoded image data"),
    ocr_type: OcrType = Form(
        OcrType.OCR,
        description="OCR processing type: 'ocr' for plain text or 'format' for structured text",
    ),
    method: Method = Form(
        Method.CHAT,
        description="Processing method: 'chat' for standard or 'chat_crop' for complex layouts",
    ),
    render: bool = Form(
        False,
        description="Whether to render formatted results as HTML (only for 'format' OCR type)",
    ),
    ocr_box: Optional[str] = Form(
        None, description="Bounding box for fine-grained OCR (format: 'x1,y1,x2,y2')"
    ),
    ocr_color: Optional[str] = Form(
        None, description="Color information for fine-grained OCR (format: 'r,g,b')"
    ),
):
    model, tokenizer = get_model_and_tokenizer()

    if by_file is None and not by_base64:
        raise HTTPException(
            status_code=400, detail="Either file or input_base64 must be provided"
        )

    if render and ocr_type != OcrType.FORMAT:
        raise HTTPException(
            status_code=400,
            detail="Render option is only available for format OCR type",
        )

    temp_dir = tempfile.mkdtemp()
    temp_file_path = None

    try:
        if by_file:
            temp_file_path = os.path.join(
                temp_dir, by_file.filename or "uploaded_image.jpg"
            )
            with open(temp_file_path, "wb") as buffer:
                shutil.copyfileobj(by_file.file, buffer)
        else:
            if by_base64 is None:
                raise ValueError("Base64 input is None")
            image_data = base64.b64decode(by_base64)
            temp_file_path = os.path.join(temp_dir, f"base64_image_{uuid.uuid4()}.jpg")
            with open(temp_file_path, "wb") as f:
                f.write(image_data)

        processor = model.chat_crop if method == Method.CHAT_CROP else model.chat
        save_render_file = (
            os.path.join(temp_dir, f"{uuid.uuid4()}.html") if render else None
        )

        kwargs = {
            "ocr_type": ocr_type,
            "render": render,
            "save_render_file": save_render_file,
        }

        if ocr_box:
            kwargs["ocr_box"] = ocr_box
        if ocr_color:
            kwargs["ocr_color"] = ocr_color

        result = processor(tokenizer, temp_file_path, **kwargs)

        if render and save_render_file and os.path.exists(save_render_file):
            with open(save_render_file, "r") as f:
                return HTMLResponse(content=f.read())

        return {"result": result}

    except Exception as e:
        if isinstance(e, HTTPException):
            raise e
        raise HTTPException(status_code=500, detail=f"Error processing image: {str(e)}")

    finally:
        if temp_dir:
            shutil.rmtree(temp_dir, ignore_errors=True)
