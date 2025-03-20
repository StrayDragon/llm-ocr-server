import typer
from typing import Optional, Literal, cast
from pathlib import Path
from core import GOTOCRProcessor

app = typer.Typer()
processor = GOTOCRProcessor()

@app.command()
def process_image(
    image_path: Path = typer.Argument(..., help="Path to input image file"),
    ocr_type: str = typer.Option("ocr", "--type", "-t", help="OCR processing type: [ocr|format]"),
    method: str = typer.Option("chat", "--method", "-m", help="Processing method: [chat|chat_crop]"),
    render: bool = typer.Option(False, "--render", "-r", help="Render formatted results"),
    output_file: Optional[Path] = typer.Option(None, "--output", "-o", help="Output file path")
):
    """Process image with GOT OCR 2.0 CPU model"""
    if render and ocr_type != "format":
        typer.echo("Error: Render option is only available for format OCR type", err=True)
        raise typer.Abort()

    # Validate and cast input types
    if ocr_type not in ("ocr", "format"):
        typer.echo(f"Error: Invalid OCR type '{ocr_type}'. Must be one of: ocr, format", err=True)
        raise typer.Abort()

    if method not in ("chat", "chat_crop"):
        typer.echo(f"Error: Invalid method '{method}'. Must be one of: chat, chat_crop", err=True)
        raise typer.Abort()

    result = processor.process_image(
        image_path,
        ocr_type=cast(Literal["ocr", "format"], ocr_type),
        method=cast(Literal["chat", "chat_crop"], method),
        render=render,
        save_render_file=output_file
    )

    if output_file:
        output_file.write_text(result)
        typer.echo(f"Results saved to {output_file}")
    else:
        typer.echo(result)

if __name__ == "__main__":
    app()
