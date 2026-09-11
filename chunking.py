import re


def chunk_text(text, chunk_size=300, overlap=100):
    """
    Chunk text from guides/manuals in a structure-aware way.

    Strategy:
    1. Split the document into paragraphs (blank-line separated — this is how
       most PDF text extractors, e.g. PyPDF/pdfplumber, naturally output text).
    2. Detect section headers (numbered headings like "3.2 Installation",
       or short ALL-CAPS lines) and keep them attached to the paragraph that
       follows, so a chunk never starts with a heading and nothing else.
    3. Greedily pack paragraphs into chunks up to chunk_size words, without
       ever splitting a paragraph or numbered step across two chunks —
       UNLESS a single paragraph alone exceeds chunk_size, in which case it
       falls back to sentence-based splitting with overlap.
    4. Adjacent chunks overlap by roughly `overlap` words, carried over from
       the end of the previous chunk, so retrieval doesn't lose context at
       chunk boundaries.

    Args:
        text: Raw extracted PDF text.
        chunk_size: Target max words per chunk.
        overlap: Approx words of overlap carried into the next chunk.

    Returns:
        List of text chunks (str).
    """
    if text is None:
        return []

    try:
        chunk_size = int(chunk_size)
        overlap = int(overlap)
    except (TypeError, ValueError) as exc:
        raise ValueError("chunk_size and overlap must be integers") from exc

    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than 0")
    if overlap < 0:
        raise ValueError("overlap must be non-negative")
    if overlap >= chunk_size:
        raise ValueError("overlap must be smaller than chunk_size")

    text = text.strip()
    if not text:
        return []

    paragraphs = _split_into_paragraphs(text)
    paragraphs = _attach_headers_to_following_paragraph(paragraphs)

    # Break any paragraph that alone exceeds chunk_size into smaller pieces
    # (falls back to word-based splitting, same idea as the original chunker).
    units = []
    for para in paragraphs:
        word_count = len(para.split())
        if word_count > chunk_size:
            units.extend(_split_oversized_paragraph(para, chunk_size, overlap))
        else:
            units.append(para)

    return _pack_units_into_chunks(units, chunk_size, overlap)


def _split_into_paragraphs(text):
    """Split on blank lines (how PDF extractors usually separate paragraphs)."""
    raw_paragraphs = re.split(r"\n\s*\n", text)
    return [p.strip() for p in raw_paragraphs if p.strip()]


HEADER_PATTERN = re.compile(
    r"^(\d+(\.\d+)*\.?\s+\S.{0,80}|[A-Z][A-Z\s]{3,60})$"
)


def _is_header(paragraph):
    """Heuristic: numbered headings ('3.2 Installation') or short ALL-CAPS lines."""
    first_line = paragraph.strip().split("\n")[0].strip()
    return bool(HEADER_PATTERN.match(first_line)) and len(first_line.split()) <= 12


def _attach_headers_to_following_paragraph(paragraphs):
    """Merge a detected header paragraph into the start of the next paragraph,
    so headers never end up alone at the tail end of a chunk with their
    content pushed into the next one."""
    merged = []
    pending_header = None

    for para in paragraphs:
        if _is_header(para) and len(para.split()) <= 12:
            pending_header = para
            continue
        if pending_header:
            merged.append(pending_header + "\n" + para)
            pending_header = None
        else:
            merged.append(para)

    if pending_header:
        merged.append(pending_header)

    return merged


def _split_oversized_paragraph(paragraph, chunk_size, overlap):
    """Fallback for a single paragraph longer than chunk_size: split on
    sentence boundaries first, then word-based as a last resort, preserving
    overlap so we don't lose context mid-explanation."""
    sentences = re.split(r"(?<=[.!?])\s+", paragraph)

    chunks = []
    current = []
    current_len = 0

    for sentence in sentences:
        sentence_len = len(sentence.split())

        # A single sentence longer than chunk_size — split by words directly.
        if sentence_len > chunk_size:
            if current:
                chunks.append(" ".join(current))
                current, current_len = [], 0
            chunks.extend(_word_split(sentence, chunk_size, overlap))
            continue

        if current_len + sentence_len > chunk_size and current:
            chunks.append(" ".join(current))
            # carry overlap words into the next chunk
            overlap_words = " ".join(current).split()[-overlap:] if overlap else []
            current = overlap_words + sentence.split()
            current_len = len(current)
        else:
            current.extend(sentence.split())
            current_len += sentence_len

    if current:
        chunks.append(" ".join(current))

    return chunks


def _word_split(text_block, chunk_size, overlap):
    """Plain fixed-size word splitting — last-resort fallback, same logic as
    the original chunker, used only for pathological long sentences."""
    words = text_block.split()
    step = chunk_size - overlap
    chunks = []
    for start in range(0, len(words), step):
        end = min(start + chunk_size, len(words))
        chunks.append(" ".join(words[start:end]))
        if end == len(words):
            break
    return chunks


def _pack_units_into_chunks(units, chunk_size, overlap):
    """Greedily combine paragraph-level units into chunks up to chunk_size
    words, carrying overlap from the end of one chunk into the next."""
    chunks = []
    current = []
    current_len = 0

    for unit in units:
        unit_len = len(unit.split())

        if current_len + unit_len > chunk_size and current:
            chunks.append("\n\n".join(current))
            # carry the last `overlap` words of the previous chunk forward
            tail_words = "\n\n".join(current).split()[-overlap:] if overlap else []
            current = [" ".join(tail_words)] if tail_words else []
            current_len = len(tail_words)

        current.append(unit)
        current_len += unit_len

    if current:
        chunks.append("\n\n".join(current))

    # Merge a tiny trailing chunk into the previous one to avoid near-empty chunks
    if len(chunks) > 1 and len(chunks[-1].split()) < overlap:
        chunks[-2] = chunks[-2] + "\n\n" + chunks[-1]
        chunks.pop()

    return chunks