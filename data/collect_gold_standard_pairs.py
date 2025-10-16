#!/usr/bin/env python3
"""
Collect gold-standard training pairs from the 4 perfect PDF-text pairs.

This script:
1. Extracts text from corrupted PDFs (garbled output due to broken ToUnicode)
2. Loads corresponding clean reference text files
3. Aligns them page-by-page to create training pairs
4. Validates data quality (input is corrupted, target is clean)

Usage:
    python3 collect_gold_standard_pairs.py \
        --pdf-dir ../corrupted_pdf_to_exctratc_text_for_comparaison \
        --txt-dir ../correct_txt \
        --output-dir data/training_pairs_gold_standard \
        --max-chunk-size 2000
"""

import argparse
import json
import logging
from pathlib import Path
from typing import List, Dict, Optional
import unicodedata
import re

# PDF extraction libraries
try:
    from pdfminer.high_level import extract_text as pdfminer_extract
    from pdfminer.layout import LAParams
except ImportError:
    print("Warning: pdfminer not installed. Run: pip install pdfminer.six")
    pdfminer_extract = None

try:
    import fitz  # PyMuPDF
except ImportError:
    print("Warning: PyMuPDF not installed. Run: pip install PyMuPDF")
    fitz = None


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# Gold standard pairs (base names without extensions)
GOLD_STANDARD_PAIRS = [
    "wiki_khmer_os",
    "ams_khmer_os_battambang",
    "lomhe_krasar",
    "cambonomist_kantumruy"
]


def has_khmer_chars(text: str) -> bool:
    """Check if text contains Khmer Unicode characters."""
    khmer_pattern = re.compile(r'[\u1780-\u17FF]')
    return bool(khmer_pattern.search(text))


def count_khmer_chars(text: str) -> int:
    """Count Khmer Unicode characters in text."""
    return sum(1 for c in text if '\u1780' <= c <= '\u17FF')


def has_pua_chars(text: str) -> bool:
    """Check if text has Private Use Area characters (sign of corruption)."""
    return any('\uE000' <= c <= '\uF8FF' for c in text)


def extract_pdf_text_pdfminer(pdf_path: Path, page_num: int) -> Optional[str]:
    """Extract text from specific PDF page using pdfminer."""
    if not pdfminer_extract:
        return None

    try:
        # Extract all pages first (pdfminer doesn't have easy per-page extraction)
        full_text = pdfminer_extract(
            str(pdf_path),
            laparams=LAParams(),
            page_numbers=[page_num]
        )
        return full_text.strip() if full_text else None
    except Exception as e:
        logger.warning(f"pdfminer extraction failed for {pdf_path} page {page_num}: {e}")
        return None


def extract_pdf_text_pymupdf(pdf_path: Path, page_num: int) -> Optional[str]:
    """Extract text from specific PDF page using PyMuPDF."""
    if not fitz:
        return None

    try:
        doc = fitz.open(str(pdf_path))
        if page_num >= len(doc):
            return None

        page = doc[page_num]
        text = page.get_text()
        doc.close()

        return text.strip() if text else None
    except Exception as e:
        logger.warning(f"PyMuPDF extraction failed for {pdf_path} page {page_num}: {e}")
        return None


def extract_pdf_text_best(pdf_path: Path, page_num: int) -> tuple[Optional[str], str]:
    """
    Extract text using best available method.
    Returns: (text, method_name)
    """
    # Try pdfminer first (usually better for corrupted PDFs)
    text = extract_pdf_text_pdfminer(pdf_path, page_num)
    if text and has_khmer_chars(text):
        return text, "pdfminer"

    # Fallback to PyMuPDF
    text = extract_pdf_text_pymupdf(pdf_path, page_num)
    if text and has_khmer_chars(text):
        return text, "pymupdf"

    return None, "none"


def split_text_into_chunks(text: str, max_chunk_size: int = 2000) -> List[str]:
    """
    Split text into chunks of roughly max_chunk_size characters.
    Tries to break at sentence boundaries (។ or newlines).
    """
    if len(text) <= max_chunk_size:
        return [text]

    chunks = []
    current_chunk = ""

    # Split by Khmer sentence ending (។) or double newline
    sentences = re.split(r'([។\n]{1,2})', text)

    for i in range(0, len(sentences), 2):
        sentence = sentences[i]
        delimiter = sentences[i+1] if i+1 < len(sentences) else ""

        if len(current_chunk) + len(sentence) + len(delimiter) > max_chunk_size:
            if current_chunk:
                chunks.append(current_chunk.strip())
                current_chunk = ""

        current_chunk += sentence + delimiter

    if current_chunk:
        chunks.append(current_chunk.strip())

    return chunks


def get_num_pages(pdf_path: Path) -> int:
    """Get number of pages in PDF."""
    if fitz:
        try:
            doc = fitz.open(str(pdf_path))
            num_pages = len(doc)
            doc.close()
            return num_pages
        except:
            pass

    # Fallback: try to extract and count
    return 100  # Assume max 100 pages


def load_reference_text(txt_path: Path) -> str:
    """Load clean reference text file."""
    try:
        with open(txt_path, 'r', encoding='utf-8') as f:
            return f.read()
    except Exception as e:
        logger.error(f"Failed to load reference text {txt_path}: {e}")
        return ""


def align_pdf_pages_to_reference(
    pdf_path: Path,
    reference_text: str,
    max_chunk_size: int = 2000,
    max_pages: int = 50
) -> List[Dict]:
    """
    Extract pages from PDF and align with reference text.

    Strategy:
    1. Extract PDF pages (corrupted text)
    2. Split reference text into chunks matching approximate PDF page lengths
    3. Create training pairs: corrupted_page → clean_chunk

    Returns list of training pairs.
    """
    pairs = []
    num_pages = min(get_num_pages(pdf_path), max_pages)

    # Extract PDF pages
    pdf_pages = []
    for page_num in range(num_pages):
        text, method = extract_pdf_text_best(pdf_path, page_num)
        if text and has_khmer_chars(text):
            pdf_pages.append({
                'text': text,
                'page_num': page_num,
                'method': method
            })
            logger.info(f"  Extracted page {page_num+1}/{num_pages} ({len(text)} chars, method: {method})")
        else:
            logger.warning(f"  Skipped page {page_num+1}/{num_pages} (no Khmer text)")

    if not pdf_pages:
        logger.warning(f"No pages extracted from {pdf_path}")
        return []

    logger.info(f"  Extracted {len(pdf_pages)} pages total")

    # Split reference text into chunks
    # Strategy: Make chunks roughly match average PDF page size
    avg_page_size = sum(len(p['text']) for p in pdf_pages) // len(pdf_pages)
    chunk_size = min(max_chunk_size, avg_page_size * 2)  # Allow some variance

    reference_chunks = split_text_into_chunks(reference_text, chunk_size)
    logger.info(f"  Split reference into {len(reference_chunks)} chunks")

    # Create pairs: PDF page → reference chunk
    # If counts don't match, align as best as possible
    num_pairs = min(len(pdf_pages), len(reference_chunks))

    for i in range(num_pairs):
        pdf_page = pdf_pages[i]
        ref_chunk = reference_chunks[i]

        # Validate pair quality
        input_khmer_ratio = count_khmer_chars(pdf_page['text']) / len(pdf_page['text']) if len(pdf_page['text']) > 0 else 0
        target_khmer_ratio = count_khmer_chars(ref_chunk) / len(ref_chunk) if len(ref_chunk) > 0 else 0

        # Skip if either has too little Khmer
        if input_khmer_ratio < 0.3 or target_khmer_ratio < 0.5:
            logger.warning(f"  Skipped pair {i+1} (low Khmer ratio: input={input_khmer_ratio:.2%}, target={target_khmer_ratio:.2%})")
            continue

        pair = {
            'input_text': pdf_page['text'],
            'target_text': ref_chunk,
            'page_num': pdf_page['page_num'],
            'pdf_name': pdf_path.name,
            'method': pdf_page['method'],
            'input_length': len(pdf_page['text']),
            'target_length': len(ref_chunk),
            'input_khmer_ratio': input_khmer_ratio,
            'target_khmer_ratio': target_khmer_ratio,
            'has_pua': has_pua_chars(pdf_page['text']),
        }

        pairs.append(pair)

    logger.info(f"  Created {len(pairs)} training pairs")
    return pairs


def collect_all_pairs(
    pdf_dir: Path,
    txt_dir: Path,
    max_chunk_size: int = 2000,
    max_pages_per_pdf: int = 50
) -> List[Dict]:
    """Collect all training pairs from gold standard corpus."""
    all_pairs = []

    for base_name in GOLD_STANDARD_PAIRS:
        pdf_path = pdf_dir / f"{base_name}.pdf"
        txt_path = txt_dir / f"{base_name}.txt"

        if not pdf_path.exists():
            logger.warning(f"PDF not found: {pdf_path}")
            continue

        if not txt_path.exists():
            logger.warning(f"Reference text not found: {txt_path}")
            continue

        logger.info(f"\nProcessing: {base_name}")
        logger.info(f"  PDF: {pdf_path}")
        logger.info(f"  Reference: {txt_path}")

        # Load reference text
        reference_text = load_reference_text(txt_path)
        if not reference_text:
            continue

        logger.info(f"  Reference length: {len(reference_text)} chars")

        # Extract and align (limit pages)
        pairs = align_pdf_pages_to_reference(pdf_path, reference_text, max_chunk_size, max_pages_per_pdf)
        all_pairs.extend(pairs)

    return all_pairs


def split_train_val_test(pairs: List[Dict], val_ratio: float = 0.1, test_ratio: float = 0.1):
    """Split pairs into train/val/test sets."""
    import random
    random.shuffle(pairs)

    n = len(pairs)
    n_test = int(n * test_ratio)
    n_val = int(n * val_ratio)
    n_train = n - n_test - n_val

    train = pairs[:n_train]
    val = pairs[n_train:n_train + n_val]
    test = pairs[n_train + n_val:]

    return train, val, test


def main():
    parser = argparse.ArgumentParser(description="Collect gold standard training pairs")
    parser.add_argument('--pdf-dir', type=str, required=True, help='Directory with corrupted PDFs')
    parser.add_argument('--txt-dir', type=str, required=True, help='Directory with clean reference texts')
    parser.add_argument('--output-dir', type=str, required=True, help='Output directory for training pairs')
    parser.add_argument('--max-chunk-size', type=int, default=2000, help='Maximum chunk size for reference text')
    parser.add_argument('--max-pages-per-pdf', type=int, default=50, help='Maximum pages to extract per PDF')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for train/val/test split')

    args = parser.parse_args()

    pdf_dir = Path(args.pdf_dir)
    txt_dir = Path(args.txt_dir)
    output_dir = Path(args.output_dir)

    if not pdf_dir.exists():
        logger.error(f"PDF directory not found: {pdf_dir}")
        return

    if not txt_dir.exists():
        logger.error(f"Text directory not found: {txt_dir}")
        return

    output_dir.mkdir(parents=True, exist_ok=True)

    # Set random seed
    import random
    random.seed(args.seed)

    logger.info(f"Collecting gold standard training pairs...")
    logger.info(f"PDF dir: {pdf_dir}")
    logger.info(f"Text dir: {txt_dir}")
    logger.info(f"Output dir: {output_dir}")
    logger.info(f"Max pages per PDF: {args.max_pages_per_pdf}")

    # Collect all pairs
    all_pairs = collect_all_pairs(pdf_dir, txt_dir, args.max_chunk_size, args.max_pages_per_pdf)

    if not all_pairs:
        logger.error("No training pairs collected!")
        return

    logger.info(f"\n✅ Collected {len(all_pairs)} total training pairs")

    # Split into train/val/test
    train, val, test = split_train_val_test(all_pairs)

    logger.info(f"\nDataset split:")
    logger.info(f"  Train: {len(train)} pairs")
    logger.info(f"  Val: {len(val)} pairs")
    logger.info(f"  Test: {len(test)} pairs")

    # Calculate statistics
    avg_input_len = sum(p['input_length'] for p in all_pairs) / len(all_pairs)
    avg_target_len = sum(p['target_length'] for p in all_pairs) / len(all_pairs)
    avg_input_khmer = sum(p['input_khmer_ratio'] for p in all_pairs) / len(all_pairs)
    avg_target_khmer = sum(p['target_khmer_ratio'] for p in all_pairs) / len(all_pairs)
    num_with_pua = sum(1 for p in all_pairs if p['has_pua'])

    logger.info(f"\nDataset statistics:")
    logger.info(f"  Avg input length: {avg_input_len:.0f} chars")
    logger.info(f"  Avg target length: {avg_target_len:.0f} chars")
    logger.info(f"  Avg input Khmer ratio: {avg_input_khmer:.2%}")
    logger.info(f"  Avg target Khmer ratio: {avg_target_khmer:.2%}")
    logger.info(f"  Pairs with PUA characters: {num_with_pua} ({num_with_pua/len(all_pairs):.1%})")

    # Save datasets
    train_file = output_dir / 'train.json'
    val_file = output_dir / 'val.json'
    test_file = output_dir / 'test.json'
    metadata_file = output_dir / 'metadata.json'

    with open(train_file, 'w', encoding='utf-8') as f:
        json.dump(train, f, ensure_ascii=False, indent=2)

    with open(val_file, 'w', encoding='utf-8') as f:
        json.dump(val, f, ensure_ascii=False, indent=2)

    with open(test_file, 'w', encoding='utf-8') as f:
        json.dump(test, f, ensure_ascii=False, indent=2)

    metadata = {
        'total_pairs': len(all_pairs),
        'train_pairs': len(train),
        'val_pairs': len(val),
        'test_pairs': len(test),
        'avg_input_length': avg_input_len,
        'avg_target_length': avg_target_len,
        'avg_input_khmer_ratio': avg_input_khmer,
        'avg_target_khmer_ratio': avg_target_khmer,
        'pairs_with_pua': num_with_pua,
        'gold_standard_files': GOLD_STANDARD_PAIRS,
        'max_chunk_size': args.max_chunk_size,
        'seed': args.seed,
    }

    with open(metadata_file, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

    logger.info(f"\n✅ Dataset saved to {output_dir}/")
    logger.info(f"  {train_file.name}: {len(train)} pairs")
    logger.info(f"  {val_file.name}: {len(val)} pairs")
    logger.info(f"  {test_file.name}: {len(test)} pairs")
    logger.info(f"  {metadata_file.name}")


if __name__ == '__main__':
    main()
