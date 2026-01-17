"""
PDF OCR script for Correspondo dataset preparation.
Extracts text from scanned PDFs (handwritten/typed letters) using EasyOCR.

Usage:
    python scripts/ocr_pdf.py input.pdf                    # Single file
    python scripts/ocr_pdf.py input_dir/ output_dir/       # Batch process directory
    python scripts/ocr_pdf.py input_dir/ --lang en fr      # Multiple languages

Requirements:
    pip install easyocr pdf2image pillow
    Also requires poppler: brew install poppler (macOS) or apt install poppler-utils (Linux)
"""

import argparse
import sys
from pathlib import Path

import easyocr
from pdf2image import convert_from_path
from PIL import Image


def ocr_image(reader: easyocr.Reader, image: Image.Image) -> str:
    """Run OCR on a single image and return extracted text."""
    results = reader.readtext(image)
    # Extract text from results, preserving reading order
    lines = [text for (_, text, _) in results]
    return '\n'.join(lines)


def process_pdf(
    pdf_path: Path,
    reader: easyocr.Reader,
    dpi: int = 300
) -> str:
    """
    Convert PDF to images and run OCR on each page.
    Returns concatenated text from all pages.
    """
    print(f"  Converting PDF to images (dpi={dpi})...")
    images = convert_from_path(str(pdf_path), dpi=dpi)

    all_text = []
    for i, image in enumerate(images):
        print(f"  Processing page {i + 1}/{len(images)}...")
        page_text = ocr_image(reader, image)
        if page_text.strip():
            all_text.append(f"--- Page {i + 1} ---\n{page_text}")

    return '\n\n'.join(all_text)


def process_single_file(
    pdf_path: Path,
    output_path: Path | None,
    reader: easyocr.Reader,
    dpi: int
) -> None:
    """Process a single PDF file."""
    print(f"\nProcessing: {pdf_path.name}")

    text = process_pdf(pdf_path, reader, dpi)

    if output_path is None:
        output_path = pdf_path.with_suffix('.txt')

    output_path.write_text(text, encoding='utf-8')
    print(f"  Saved: {output_path}")
    print(f"  Extracted {len(text):,} characters")


def process_directory(
    input_dir: Path,
    output_dir: Path,
    reader: easyocr.Reader,
    dpi: int
) -> None:
    """Process all PDFs in a directory."""
    output_dir.mkdir(parents=True, exist_ok=True)

    pdf_files = list(input_dir.glob('*.pdf')) + list(input_dir.glob('*.PDF'))

    if not pdf_files:
        print(f"No PDF files found in {input_dir}")
        return

    print(f"Found {len(pdf_files)} PDF files")

    for pdf_path in sorted(pdf_files):
        output_path = output_dir / pdf_path.with_suffix('.txt').name
        process_single_file(pdf_path, output_path, reader, dpi)

    print(f"\nCompleted! Processed {len(pdf_files)} files -> {output_dir}")


def main():
    parser = argparse.ArgumentParser(
        description='Extract text from PDFs using OCR (EasyOCR)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument(
        'input',
        type=Path,
        help='Input PDF file or directory containing PDFs'
    )
    parser.add_argument(
        'output',
        type=Path,
        nargs='?',
        default=None,
        help='Output text file or directory (default: same as input with .txt extension)'
    )
    parser.add_argument(
        '--lang',
        nargs='+',
        default=['en'],
        help='Language(s) for OCR (default: en). Examples: en, fr, de, es'
    )
    parser.add_argument(
        '--dpi',
        type=int,
        default=300,
        help='DPI for PDF to image conversion (default: 300, higher = better quality but slower)'
    )
    parser.add_argument(
        '--gpu',
        action='store_true',
        default=True,
        help='Use GPU acceleration (default: True)'
    )
    parser.add_argument(
        '--no-gpu',
        action='store_true',
        help='Disable GPU, use CPU only'
    )

    args = parser.parse_args()

    # Validate input
    if not args.input.exists():
        print(f"Error: Input path does not exist: {args.input}", file=sys.stderr)
        sys.exit(1)

    # Initialize EasyOCR reader
    use_gpu = args.gpu and not args.no_gpu
    print(f"Initializing EasyOCR (languages: {args.lang}, GPU: {use_gpu})...")
    reader = easyocr.Reader(args.lang, gpu=use_gpu)

    # Process input
    if args.input.is_file():
        process_single_file(args.input, args.output, reader, args.dpi)
    elif args.input.is_dir():
        output_dir = args.output if args.output else args.input / 'ocr_output'
        process_directory(args.input, output_dir, reader, args.dpi)
    else:
        print(f"Error: Invalid input path: {args.input}", file=sys.stderr)
        sys.exit(1)


if __name__ == '__main__':
    main()
