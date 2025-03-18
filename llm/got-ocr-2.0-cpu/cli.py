import typer
from typing import Optional
from pathlib import Path
from transformers import AutoModel, AutoTokenizer

app = typer.Typer()

def load_model():
    tokenizer = AutoTokenizer.from_pretrained('srimanth-d/GOT_CPU', trust_remote_code=True)
    model = AutoModel.from_pretrained('srimanth-d/GOT_CPU',
                                    trust_remote_code=True,
                                    low_cpu_mem_usage=True,
                                    use_safetensors=True,
                                    pad_token_id=tokenizer.eos_token_id)
    return model.eval(), tokenizer

@app.command()
def process_image(
    image_path: Path = typer.Argument(..., help="Path to input image file"),
    ocr_type: str = typer.Option("ocr", "--type", "-t", help="OCR processing type: [ocr|format]"),
    method: str = typer.Option("chat", "--method", "-m", help="Processing method: [chat|chat_crop]"),
    render: bool = typer.Option(False, "--render", "-r", help="Render formatted results"),
    output_file: Optional[Path] = typer.Option(None, "--output", "-o", help="Output file path")
):
    """Process image with GOT OCR 2.0 CPU model"""
    model, tokenizer = load_model()

    if render and ocr_type != "format":
        typer.echo("Error: Render option is only available for format OCR type", err=True)
        raise typer.Abort()

    # Choose processing method
    if method == "crop":
        processor = model.chat_crop
    else:
        processor = model.chat

    # Process the image
    result = processor(
        tokenizer,
        str(image_path),
        ocr_type=ocr_type,
        render=render,
        save_render_file=str(output_file) if output_file else None
    )

    if output_file:
        output_file.write_text(result)
        typer.echo(f"Results saved to {output_file}")
    else:
        typer.echo(result)

if __name__ == "__main__":
    app()
