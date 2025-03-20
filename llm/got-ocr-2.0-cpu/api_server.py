import time
from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.concurrency import asynccontextmanager
import base64
from typing import Optional
from enum import Enum
import shutil
from io import BytesIO
from pydantic import BaseModel, Field
from bookxnote_local_ocr_openapi.server_sdk.service.service import (
    AbstractRootService as AbstractBookxnoteLocalOCRService,
)
from bookxnote_local_ocr_openapi.server_sdk import ApiAuth as BookxnoteLocalOCRApiAuth
from bookxnote_local_ocr_openapi.server_sdk.types.post_ocr_by_bxn_local_ocr_response import (
    PostOcrByBxnLocalOcrResponse,
)
from bookxnote_local_ocr_openapi.server_sdk.types.post_ocr_by_bxn_local_ocr_response_data import (
    PostOcrByBxnLocalOcrResponseData,
)
from bookxnote_local_ocr_openapi.server_sdk.errors import (
    BadRequestError as BxnLocalOCRBadRequestError,
)
from bookxnote_local_ocr_openapi.server_sdk.types.bad_request_error_body import (
    BadRequestErrorBody as BxnLocalOCRBadRequestErrorBody,
)
from bookxnote_local_ocr_openapi.server_sdk.register import (
    register as register_bookxnote_local_ocr,
)

from core import GOTOCRProcessor


processor = GOTOCRProcessor()


@asynccontextmanager
async def lifespan(_: FastAPI):
    processor.get_model_and_tokenizer()  # Preload model
    yield
    processor.clear_cache()


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


def _process_image(
    *,
    by_file: UploadFile | None = None,
    by_base64: str | None = None,
    ocr_type: OcrType,
    method: Method,
    render: bool = False,
    ocr_box: str | None = None,
    ocr_color: str | None = None,
) -> str:
    image_buffer = BytesIO()
    if by_file:
        shutil.copyfileobj(by_file.file, image_buffer)
    else:
        if by_base64 is None:
            raise ValueError("Base64 input is None")
        image_buffer.write(base64.b64decode(by_base64))

    image_buffer.seek(0)

    result = processor.process_image(
        image_buffer,
        ocr_type=ocr_type.value,
        method=method.value,
        render=render,
        ocr_box=ocr_box,
        ocr_color=ocr_color,
    )
    return result


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
    if by_file is None and not by_base64:
        raise HTTPException(
            status_code=400, detail="Either file or input_base64 must be provided"
        )

    if render and ocr_type != OcrType.FORMAT:
        raise HTTPException(
            status_code=400,
            detail="Render option is only available for format OCR type",
        )

    try:
        result = _process_image(
            by_file=by_file,
            by_base64=by_base64,
            ocr_type=ocr_type,
            method=method,
            render=render,
            ocr_box=ocr_box,
            ocr_color=ocr_color,
        )

        return {"result": result}
    except Exception as e:
        if isinstance(e, HTTPException):
            raise e
        raise HTTPException(status_code=500, detail=f"Error processing image: {str(e)}")


class BookxnoteLocalOCRService(AbstractBookxnoteLocalOCRService):
    def post_ocr_by_bxn_local_ocr(
        self,
        *,
        base64_image: str | None = Form(None),
        auth: BookxnoteLocalOCRApiAuth,
    ) -> PostOcrByBxnLocalOcrResponse:
        # skip auth
        _ = auth

        if not base64_image:
            raise BxnLocalOCRBadRequestError(
                error=BxnLocalOCRBadRequestErrorBody(
                    code=400,
                    msg="base64_image is required and must non empty",
                )
            )

        bg = time.time()
        try:
            result = _process_image(
                by_base64=base64_image,
                ocr_type=OcrType.OCR,
                method=Method.CHAT_CROP,
            )
        except Exception as e:
            raise BxnLocalOCRBadRequestError(
                error=BxnLocalOCRBadRequestErrorBody(
                    code=400,
                    msg=f"Error processing image: {e!s}",
                )
            )
        ed = time.time()

        return PostOcrByBxnLocalOcrResponse(
            code=0,
            msg="success",
            data=PostOcrByBxnLocalOcrResponseData(
                text=result,
                confidence=1.0,
                time_cost=ed - bg,
            ),
        )


register_bookxnote_local_ocr(app, root=BookxnoteLocalOCRService())
