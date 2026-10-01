from app.ingestion.chunker import chunk_pages
from app.ingestion.quality import assess
from app.text import bm25_document_vector, bm25_query_vector, is_devanagari, normalize, tokenize

GARBLED = (
    "ऐसा बहुत कम होता है चक हमारे मातंग समाज में चकसी के मन में चकसी के प्रचत घृणा की भावना आये। "
    "लेचकन उस शाम हवा में तनाव तैर रहा था क्ोंचक दशकों बाद हम घृणा देख रहे थे। िर िहिान िल "
) * 5
CLEAN = (
    "भगवान् हनुमान ने पूछा कि आपके मन में उन लोगों के प्रति कटु भावनाएं क्यों हैं जो जंगल से बाहर "
    "रहते हैं? उन्होंने कहा कि यह दिन बहुत शुभ है और फिर सबने मिलकर किया। "
) * 5


def test_quality_detects_broken_font_mapping():
    assert assess(GARBLED).needs_ocr
    assert not assess(CLEAN).needs_ocr
    assert assess("").needs_ocr
    assert not assess("This is an English page about the Bhagavad Gita and karma yoga. " * 3).needs_ocr


def test_normalize_converts_pipe_danda_and_whitespace():
    assert normalize("राम वन गए |  सीता   साथ थीं ||") == "राम वन गए । सीता साथ थीं ॥"


def test_tokenize_drops_stopwords_and_keeps_names():
    toks = tokenize("हनुमान जी कौन हैं? Who is Hanuman")
    assert "हनुमान" in toks and "hanuman" in toks
    assert "हैं" not in toks and "is" not in toks


def test_is_devanagari():
    assert is_devanagari("हनुमान जी कौन हैं?")
    assert not is_devanagari("who is hanuman")


def test_bm25_vectors_share_ids():
    d_idx, d_val = bm25_document_vector("हनुमान ने लंका जलाई हनुमान", avg_len=10)
    q_idx, _ = bm25_query_vector("हनुमान")
    assert set(q_idx) <= set(d_idx)
    assert all(v > 0 for v in d_val)


def test_chunker_respects_sections_size_and_pages():
    pages = [
        (1, "## अध्याय एक\n\n" + "राम वन गए। " * 30),
        (2, "सीता साथ थीं। " * 30 + "\n\n## अध्याय दो\n\nहनुमान लंका गए।"),
    ]
    chunks = chunk_pages(pages, size=200, overlap=40)
    assert all(len(c.text) <= 200 for c in chunks)
    assert {c.section for c in chunks} == {"अध्याय एक", "अध्याय दो"}
    assert any(c.page_start == 1 and c.page_end == 2 for c in chunks)  # spans page break
    assert chunks[-1].section == "अध्याय दो" and chunks[-1].page_start == 2
    assert [c.index for c in chunks] == list(range(len(chunks)))


def test_chunker_terminates_on_huge_sentence():
    chunks = chunk_pages([(1, "शब्द " * 1000)], size=100, overlap=50)
    assert chunks and all(len(c.text) <= 100 for c in chunks)
