"""Clean, deduplicate, and split FIT-HCMUTE chunks into focused corpora."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any, Iterable


CORE_CATEGORIES = {
    "introduction",
    "program",
    "admission",
    "graduate",
    "department",
    "research",
}

CATEGORY_KEYWORDS: dict[str, tuple[str, ...]] = {
    "introduction": (
        "giới thiệu khoa",
        "giới thiệu về khoa",
        "lịch sử",
        "thành lập",
        "được thành lập",
        "khoa công nghệ thông tin được thành lập",
        "khoa cntt được thành lập",
        "hình thành và phát triển",
    ),
    "program": (
        "chương trình đào tạo",
        "công nghệ thông tin",
        "an toàn thông tin",
        "kỹ thuật dữ liệu",
        "chuan dau ra",
        "chuẩn đầu ra",
        "mục tiêu đào tạo",
        "7480201",
        "7480202",
    ),
    "admission": (
        "tuyển sinh",
        "xét tuyển",
        "phương thức tuyển sinh",
        "chỉ tiêu",
        "đề án tuyển sinh",
        "đăng ký xét tuyển",
    ),
    "graduate": (
        "thạc sĩ",
        "sau đại học",
        "khoa học máy tính",
        "8480101",
        "cao học",
    ),
    "department": (
        "bộ môn",
        "công nghệ phần mềm",
        "hệ thống thông tin",
        "mạng và an ninh mạng",
        "trí tuệ nhân tạo",
        "bộ môn công nghệ phần mềm",
        "bộ môn hệ thống thông tin",
    ),
    "research": (
        "nghiên cứu",
        "đề tài",
        "hội nghị nghiên cứu khoa học",
        "seminar",
        "publication",
        "công bố khoa học",
        "bài báo khoa học",
    ),
    "news_event": (
        "olympic",
        "icpc",
        "workshop",
        "netcompany",
        "gameloft",
        "tuyển dụng",
        "học bổng",
        "thông báo",
        "sự kiện",
        "cuộc thi",
        "đăng ký tham gia",
        "internship",
        "ngày hội",
        "thực tập sinh",
        "career tour",
        "fresher",
    ),
}

NOISE_KEYWORDS = (
    "trang chủ",
    "đăng nhập",
    "copyright",
    "facebook",
    "youtube",
    "sitemap",
    "menu",
    "liên hệ",
    "email",
    "website",
)

PROGRAM_ANCHORS = (
    "chương trình đào tạo",
    "chuẩn đầu ra",
    "chuan dau ra",
    "mục tiêu đào tạo",
    "7480201",
    "7480202",
    "đào tạo ngành",
    "ngành công nghệ thông tin",
    "ngành an toàn thông tin",
    "ngành kỹ thuật dữ liệu",
    "khối kiến thức",
    "tín chỉ",
)

STRONG_CORE_ANCHORS: dict[str, tuple[str, ...]] = {
    "introduction": ("giới thiệu", "lịch sử", "thành lập", "hình thành và phát triển"),
    "program": PROGRAM_ANCHORS,
    "admission": ("tuyển sinh", "xét tuyển", "phương thức tuyển sinh", "chỉ tiêu"),
    "graduate": ("thạc sĩ", "sau đại học", "8480101", "cao học"),
    "department": ("bộ môn",),
    "research": ("đề tài", "hội nghị nghiên cứu khoa học", "publication", "công bố khoa học", "bài báo khoa học"),
}

INTRODUCTION_PRIORITY_ANCHORS = (
    "giới thiệu khoa",
    "gioi-thieu-khoa",
    "khoa công nghệ thông tin được thành lập",
    "được thành lập",
    "thành lập năm",
    "lịch sử",
    "hình thành và phát triển",
)


def normalize_text(text: Any) -> str:
    """Normalize whitespace while preserving Vietnamese Unicode."""
    if text is None:
        return ""
    text_value = str(text).replace("\u00a0", " ").replace("\ufeff", " ")
    return re.sub(r"\s+", " ", text_value).strip()


def normalized_hash(text: str) -> str:
    normalized = re.sub(r"\s+", " ", text.casefold()).strip()
    return hashlib.sha1(normalized.encode("utf-8")).hexdigest()


def stable_id(*parts: Any, prefix: str = "auto") -> str:
    source = "||".join(normalize_text(part) for part in parts if part is not None)
    if not source:
        source = prefix
    digest = hashlib.sha1(source.encode("utf-8")).hexdigest()[:16]
    return f"{prefix}_{digest}"


def count_words(text: str) -> int:
    return len(re.findall(r"\S+", text))


def keyword_hits(text: str, keywords: Iterable[str]) -> tuple[int, int]:
    text_lc = text.casefold()
    total = 0
    unique = 0
    for keyword in keywords:
        hits = text_lc.count(keyword.casefold())
        if hits:
            total += hits
            unique += 1
    return total, unique


def is_noise(text: str, min_chars: int) -> tuple[bool, str]:
    if len(text) < min_chars:
        return True, "too_short"

    text_lc = text.casefold()
    words = max(count_words(text), 1)

    url_count = len(re.findall(r"https?://|www\.|href=", text_lc))
    if url_count >= 3 or (url_count >= 2 and url_count / words > 0.05):
        return True, "too_many_links"

    pipe_count = text.count("|")
    if pipe_count >= 8 or (pipe_count >= 4 and pipe_count / max(len(text), 1) > 0.025):
        return True, "too_many_pipes"

    noise_total, noise_unique = keyword_hits(text_lc, NOISE_KEYWORDS)
    nav_phrase_count = len(
        re.findall(
            r"\b(home|login|logout|search|copyright|sitemap|menu)\b",
            text_lc,
        )
    )

    if noise_unique >= 5 or noise_total >= 8:
        return True, "many_noise_keywords"
    if len(text) < 350 and noise_unique >= 3 and noise_total + nav_phrase_count >= 4:
        return True, "navigation_footer"
    if len(text) < 260 and noise_unique >= 2 and pipe_count >= 3:
        return True, "navigation_footer"

    return False, ""


def has_any(text: str, keywords: Iterable[str]) -> bool:
    text_lc = text.casefold()
    return any(keyword.casefold() in text_lc for keyword in keywords)


def category_score(text: str, title: str, url: str, category: str) -> int:
    keywords = CATEGORY_KEYWORDS[category]
    text_total, text_unique = keyword_hits(text, keywords)
    title_total, title_unique = keyword_hits(title, keywords)
    url_total, url_unique = keyword_hits(url, keywords)
    return (text_total * 3) + (text_unique * 4) + (title_total * 1) + (title_unique * 2) + url_total + url_unique


def classify_category(text: str, title: str = "", url: str = "") -> str:
    text_lc = text.casefold()
    title_lc = title.casefold()
    url_lc = url.casefold()
    combined_core_context = f"{title_lc} {text_lc}"

    scores: dict[str, int] = {}
    for category, keywords in CATEGORY_KEYWORDS.items():
        score = category_score(text_lc, title_lc, url_lc, category)
        if category == "program" and score and not has_any(combined_core_context, PROGRAM_ANCHORS):
            score = 0
        if score:
            scores[category] = score

    if not scores:
        return "general"

    if scores.get("introduction") and has_any(combined_core_context, INTRODUCTION_PRIORITY_ANCHORS):
        return "introduction"

    news_score = scores.get("news_event", 0)
    core_scores = {key: scores[key] for key in CORE_CATEGORIES if key in scores}
    if news_score:
        best_core_category = ""
        best_core_score = 0
        if core_scores:
            best_core_category, best_core_score = max(core_scores.items(), key=lambda item: item[1])
        strong_core = bool(
            best_core_category
            and has_any(combined_core_context, STRONG_CORE_ANCHORS.get(best_core_category, ()))
        )
        if news_score >= 8 and (not strong_core or news_score >= best_core_score):
            return "news_event"

    if core_scores:
        return max(core_scores.items(), key=lambda item: item[1])[0]
    if "news_event" in scores:
        return "news_event"
    return "general"


def corpus_type_for(category: str) -> str:
    if category in CORE_CATEGORIES:
        return "core"
    if category == "news_event":
        return "news"
    return "general"


def sample_record(line_no: int, reason: str, raw_line: str | None = None, record: dict[str, Any] | None = None) -> dict[str, Any]:
    sample: dict[str, Any] = {"line_no": line_no, "reason": reason}
    if raw_line is not None:
        sample["raw_line"] = raw_line.rstrip("\n")
    if record is not None:
        sample["chunk_id"] = record.get("chunk_id")
        sample["doc_id"] = record.get("doc_id")
        sample["title"] = record.get("title")
        sample["url"] = record.get("url")
        sample["text"] = record.get("text")
    return sample


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as file:
        for row in rows:
            file.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_json(path: Path, payload: dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, ensure_ascii=False, indent=2)
        file.write("\n")


def title_samples(rows: list[dict[str, Any]], limit: int = 10) -> list[str]:
    titles: list[str] = []
    seen: set[str] = set()
    for row in rows:
        title = normalize_text(row.get("title"))
        if title and title not in seen:
            titles.append(title)
            seen.add(title)
        if len(titles) >= limit:
            break
    return titles


def clean_and_split(input_path: Path, out_dir: Path, min_chars: int) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)

    kept: list[dict[str, Any]] = []
    core_rows: list[dict[str, Any]] = []
    news_rows: list[dict[str, Any]] = []
    general_rows: list[dict[str, Any]] = []
    noisy_samples: list[dict[str, Any]] = []
    duplicate_samples: list[dict[str, Any]] = []
    invalid_samples: list[dict[str, Any]] = []
    seen_hashes: set[str] = set()

    total_input_lines = 0
    valid_json_lines = 0

    with input_path.open("r", encoding="utf-8") as file:
        for line_no, line in enumerate(file, start=1):
            total_input_lines += 1
            if not line.strip():
                continue

            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                invalid_samples.append(sample_record(line_no, f"json_decode_error: {exc.msg}", raw_line=line))
                continue

            if not isinstance(row, dict):
                invalid_samples.append(sample_record(line_no, "json_value_is_not_object", raw_line=line))
                continue

            valid_json_lines += 1
            cleaned_text = normalize_text(row.get("text"))
            record = dict(row)
            record["text"] = cleaned_text

            noisy, reason = is_noise(cleaned_text, min_chars=min_chars)
            if noisy:
                noisy_samples.append(sample_record(line_no, reason, record=record))
                continue

            text_hash = normalized_hash(cleaned_text)
            if text_hash in seen_hashes:
                duplicate_samples.append(sample_record(line_no, "duplicate_normalized_text", record=record))
                continue
            seen_hashes.add(text_hash)

            title = normalize_text(record.get("title"))
            url = normalize_text(record.get("url"))
            if not normalize_text(record.get("chunk_id")):
                record["chunk_id"] = stable_id(url, title, cleaned_text, prefix="chunk")
            if not normalize_text(record.get("doc_id")):
                record["doc_id"] = stable_id(url, title, prefix="doc")

            category = classify_category(cleaned_text, title=title, url=url)
            corpus_type = corpus_type_for(category)
            record["category_clean"] = category
            record["corpus_type"] = corpus_type
            record["text_chars"] = len(cleaned_text)
            record["text_words"] = count_words(cleaned_text)

            kept.append(record)
            if corpus_type == "core":
                core_rows.append(record)
            elif corpus_type == "news":
                news_rows.append(record)
            else:
                general_rows.append(record)

    output_files = {
        "core": out_dir / "chunks_core_clean.jsonl",
        "news": out_dir / "chunks_news_clean.jsonl",
        "general": out_dir / "chunks_general_clean.jsonl",
        "all": out_dir / "chunks_filtered_all.jsonl",
        "noisy": out_dir / "noisy_dropped_samples.jsonl",
        "duplicates": out_dir / "duplicate_dropped_samples.jsonl",
        "invalid": out_dir / "invalid_json_samples.jsonl",
    }

    write_jsonl(output_files["core"], core_rows)
    write_jsonl(output_files["news"], news_rows)
    write_jsonl(output_files["general"], general_rows)
    write_jsonl(output_files["all"], kept)
    write_jsonl(output_files["noisy"], noisy_samples)
    write_jsonl(output_files["duplicates"], duplicate_samples)
    write_jsonl(output_files["invalid"], invalid_samples)

    category_distribution = Counter(row["category_clean"] for row in kept)
    corpus_type_distribution = Counter(row["corpus_type"] for row in kept)
    title_counts = Counter(normalize_text(row.get("title")) or "(missing title)" for row in kept)
    report = {
        "input_path": str(input_path),
        "output_dir": str(out_dir),
        "total_input_lines": total_input_lines,
        "valid_json_lines": valid_json_lines,
        "kept_chunks": len(kept),
        "core_chunks": len(core_rows),
        "news_chunks": len(news_rows),
        "general_chunks": len(general_rows),
        "dropped_noise": len(noisy_samples),
        "dropped_duplicates": len(duplicate_samples),
        "invalid_json_lines": len(invalid_samples),
        "category_distribution": dict(category_distribution),
        "corpus_type_distribution": dict(corpus_type_distribution),
        "top_titles_by_count": [
            {"title": title, "count": count} for title, count in title_counts.most_common(20)
        ],
        "sample_core_titles": title_samples(core_rows),
        "sample_news_titles": title_samples(news_rows),
        "sample_general_titles": title_samples(general_rows),
    }
    write_json(out_dir / "corpus_report.json", report)
    return report


def print_summary(report: dict[str, Any]) -> None:
    print("Corpus cleaning summary")
    print(f"  input_path: {report['input_path']}")
    print(f"  output_dir: {report['output_dir']}")
    print(f"  total_input_lines: {report['total_input_lines']}")
    print(f"  valid_json_lines: {report['valid_json_lines']}")
    print(f"  kept_chunks: {report['kept_chunks']}")
    print(f"  core_chunks: {report['core_chunks']}")
    print(f"  news_chunks: {report['news_chunks']}")
    print(f"  general_chunks: {report['general_chunks']}")
    print(f"  dropped_noise: {report['dropped_noise']}")
    print(f"  dropped_duplicates: {report['dropped_duplicates']}")
    print(f"  invalid_json_lines: {report['invalid_json_lines']}")


def run_self_test() -> None:
    intro_text = (
        "Giới thiệu Khoa Công nghệ Thông tin HCMUTE. "
        "Khoa Công nghệ Thông tin được thành lập nhằm đào tạo nguồn nhân lực chất lượng cao "
        "trong lĩnh vực máy tính và công nghệ thông tin cho xã hội."
    )
    program_text = (
        "Chương trình đào tạo ngành Công nghệ thông tin mã ngành 7480201 mô tả mục tiêu đào tạo, "
        "chuẩn đầu ra, kiến thức nền tảng, kỹ năng nghề nghiệp và định hướng phát triển cho sinh viên."
    )
    news_text = (
        "Thông báo workshop Netcompany dành cho sinh viên FIT HCMUTE. Chương trình có đăng ký tham gia, "
        "chia sẻ cơ hội internship, tuyển dụng và giao lưu với doanh nghiệp trong ngày hội nghề nghiệp."
    )
    noisy_text = (
        "Trang chủ | Menu | Facebook | Youtube | Sitemap | Đăng nhập | Liên hệ | Email | Website | "
        "Copyright FIT HCMUTE"
    )

    rows = [
        {"title": "Giới thiệu khoa", "url": "https://fit.hcmute.edu.vn/gioi-thieu", "text": intro_text},
        {"title": "Chương trình đào tạo", "url": "https://fit.hcmute.edu.vn/dao-tao", "text": program_text},
        {"title": "Workshop Netcompany", "url": "https://fit.hcmute.edu.vn/tin-tuc", "text": news_text},
        {"title": "Menu", "url": "https://fit.hcmute.edu.vn", "text": noisy_text},
        {"title": "Duplicate intro", "url": "https://fit.hcmute.edu.vn/dup", "text": intro_text},
    ]

    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        input_path = tmp_path / "self_test.jsonl"
        out_dir = tmp_path / "processed"
        with input_path.open("w", encoding="utf-8") as file:
            for row in rows:
                file.write(json.dumps(row, ensure_ascii=False) + "\n")
            file.write("{invalid json line\n")

        report = clean_and_split(input_path=input_path, out_dir=out_dir, min_chars=80)

        assert report["core_chunks"] >= 1, "expected at least one core chunk"
        assert report["news_chunks"] >= 1, "expected at least one news chunk"
        assert report["dropped_noise"] >= 1, "expected dropped noise"
        assert report["dropped_duplicates"] >= 1, "expected dropped duplicate"
        assert report["invalid_json_lines"] >= 1, "expected invalid JSON"
        print("SELF TEST PASSED")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Clean and split FIT-HCMUTE JSONL chunks.")
    parser.add_argument("--input", default="data/chunks_clean.jsonl", help="Input JSONL path.")
    parser.add_argument("--out-dir", default="data/processed", help="Output directory.")
    parser.add_argument("--min-chars", type=int, default=120, help="Minimum cleaned text characters to keep.")
    parser.add_argument("--self-test", action="store_true", help="Run built-in self-test.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.self_test:
        run_self_test()
        return

    input_path = Path(args.input)
    out_dir = Path(args.out_dir)
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    report = clean_and_split(input_path=input_path, out_dir=out_dir, min_chars=args.min_chars)
    print_summary(report)


if __name__ == "__main__":
    main()
