"""Multi-format fault-tolerant resume parser (PDF, DOCX, TXT) with layout-aware text, font-size name detection, and hyperlink extraction."""

from __future__ import annotations

import hashlib
import logging
import re
import sys
import types
import zipfile
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Defensive compatibility shim in case user site-packages has an incomplete cryptography install
try:
    from cryptography.hazmat.backends import default_backend  # type: ignore
except Exception:
    for _mod in (
        "cryptography",
        "cryptography.hazmat",
        "cryptography.hazmat.backends",
        "cryptography.hazmat.primitives",
        "cryptography.hazmat.primitives.ciphers",
        "cryptography.hazmat.primitives.ciphers.algorithms",
        "cryptography.hazmat.primitives.ciphers.modes",
    ):
        if _mod not in sys.modules:
            sys.modules[_mod] = types.ModuleType(_mod)
    sys.modules["cryptography"].__version__ = "42.0.0"  # type: ignore
    sys.modules["cryptography.hazmat.backends"].default_backend = lambda: None  # type: ignore
    sys.modules["cryptography.hazmat.primitives.ciphers"].Cipher = None  # type: ignore
    sys.modules["cryptography.hazmat.primitives.ciphers.algorithms"].AES = None  # type: ignore
    sys.modules["cryptography.hazmat.primitives.ciphers.algorithms"].ARC4 = None  # type: ignore
    sys.modules["cryptography.hazmat.primitives.ciphers.modes"].CBC = None  # type: ignore
    sys.modules["cryptography.hazmat.primitives.ciphers.modes"].ECB = None  # type: ignore

import pdfplumber
from pdfminer.high_level import extract_text as pdfminer_extract_text

from src.config import SKILL_PATTERNS
from src.models import ParsedResume

logging.getLogger("pdfminer").setLevel(logging.ERROR)

SUPPORTED_EXTENSIONS = {".pdf", ".docx", ".txt"}

RESERVED_GITHUB_PATHS = {
    "",
    "about",
    "collections",
    "contact",
    "customer-stories",
    "enterprise",
    "events",
    "explore",
    "features",
    "gist",
    "git-guides",
    "join",
    "login",
    "marketplace",
    "Mobile",
    "new",
    "notifications",
    "orgs",
    "pricing",
    "readme",
    "search",
    "security",
    "settings",
    "signup",
    "site",
    "sponsors",
    "team",
    "topics",
    "trending",
    "users",
}

ROLE_STOPWORDS = {
    "ai",
    "ml",
    "ai/ml",
    "engineer",
    "developer",
    "software",
    "full",
    "stack",
    "fullstack",
    "full-stack",
    "backend",
    "frontend",
    "intern",
    "associate",
    "aspiring",
    "summary",
    "professional",
    "objective",
    "profile",
    "education",
    "experience",
    "skills",
    "projects",
    "generative",
    "data",
    "scientist",
    "analyst",
    "bengaluru",
    "bangalore",
    "hyderabad",
    "india",
    "chennai",
    "delhi",
    "pune",
    "mumbai",
    "jaipur",
    "karnataka",
    "linkedin",
    "github",
    "leetcode",
    "portfolio",
    "email",
    "phone",
    "mobile",
}

SECTION_HEADERS_MAP: List[Tuple[str, str]] = [
    ("summary", r"^(?:professional\s+summary|profile\s+summary|resume\s+summary|executive\s+summary|summary|profile\s+objective|career\s+aspiration|career\s+objective|objectives?|profile|about\s+me)\s*$"),
    ("experience", r"^(?:work\s+experience|professional\s+experience|experience|internships?|employment\s+history|hackathon\s+experience)\s*$"),
    ("projects", r"^(?:key\s+projects|selected\s+projects|project\s+experience|academic\s+projects|personal\s+projects|major\s+projects|projects|open\s+source(?:\s+contributions?)?)\s*$"),
    ("skills", r"^(?:technical\s+skills|core\s+technical\s+expertise|skills\s+summary|skills\s*&\s*interests|skills\s*&\s*tools|skills|technologies|core\s+competencies|ats\s+keyword\s+index)\s*$"),
    ("education", r"^(?:education|academic\s+background|qualifications)\s*$"),
    ("certifications", r"^(?:certifications?(?:\s*&\s*achievements)?|certificates(?:\s*&\s*achievements)?|licenses(?:\s*&\s*certifications?)?)\s*$"),
    ("achievements", r"^(?:key\s+achievements(?:\s*&\s*metrics)?|achievements?(?:\s*&\s*leadership)?|awards(?:\s*&\s*honors)?|honors|publications?|research\s*\(under\s+review\)|extra[\s-]+curricular\s+activities|co-curricular\s+activities)\s*$"),
]

PLACEHOLDER_EMAIL_DOMAINS = {"email.com", "example.com", "domain.com", "yourmail.com", "test.com"}


def compute_file_hash(file_path: Path) -> str:
    """Compute SHA-256 hash of a file for duplicate detection."""
    hasher = hashlib.sha256()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def extract_email(text: str, hyperlinks: List[str]) -> Optional[str]:
    """Extract candidate email from mailto hyperlinks or resume text, preferring real domains over template placeholders."""
    candidates: List[str] = []
    for uri in hyperlinks:
        if uri.lower().startswith("mailto:"):
            addr = uri[7:].split("?")[0].strip()
            if "@" in addr and addr not in candidates:
                candidates.append(addr)
    for m in re.finditer(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}", text):
        addr = m.group(0).strip()
        if addr not in candidates:
            candidates.append(addr)
    for addr in candidates:
        domain = addr.rsplit("@", 1)[-1].lower()
        if domain not in PLACEHOLDER_EMAIL_DOMAINS:
            return addr
    return candidates[0] if candidates else None


def extract_phone(text: str) -> Optional[str]:
    """Extract phone number from resume text."""
    match = re.search(
        r"(?:\+91[\s-]?)?[6-9]\d{4}[\s-]?\d{5}\b|(?:\+91[\s-]?)?[6-9]\d{2}[\s-]?\d{3}[\s-]?\d{4}\b",
        text,
    )
    return match.group(0).strip() if match else None


def extract_github_info(
    raw_text: str, plumber_text: str, hyperlinks: List[str]
) -> Tuple[Optional[str], Optional[str], List[str]]:
    """
    Extract GitHub profile URL, username, and any repo URLs from hyperlinks and text.
    Prioritizes embedded PDF hyperlink annotations to avoid line-wrap truncation.
    """
    candidate_usernames: List[str] = []
    repo_urls: List[str] = []

    def _parse_gh_url(url_str: str) -> None:
        cleaned = url_str.strip().rstrip(".,;)|/ ")
        m = re.search(
            r"(?:https?://)?(?:www\.)?github\.com/([A-Za-z0-9-]+)(?:/([A-Za-z0-9_.-]+))?",
            cleaned,
            re.IGNORECASE,
        )
        if not m:
            return
        user = m.group(1).strip("-")
        repo = m.group(2)
        if not user or user.lower() in RESERVED_GITHUB_PATHS:
            return
        if user not in candidate_usernames:
            candidate_usernames.append(user)
        if repo and repo.lower() not in ("repositories", "stars", "followers"):
            full_repo = f"https://github.com/{user}/{repo}"
            if full_repo not in repo_urls:
                repo_urls.append(full_repo)

    # 1. Check PDF hyperlinks first (most reliable, never truncated by PDF text wrapping)
    for uri in hyperlinks:
        if "github.com" in uri.lower():
            _parse_gh_url(uri)

    # 2. Check extracted text (both miner and plumber text)
    for txt in (plumber_text, raw_text):
        for m in re.finditer(
            r"(?:https?://)?(?:www\.)?github\.com/[A-Za-z0-9-]+(?:/[A-Za-z0-9_.-]+)?",
            txt,
            re.IGNORECASE,
        ):
            _parse_gh_url(m.group(0))

    # 3. Fallback: explicit "GitHub: @username" or "GitHub: username" in text when no github.com URL is present
    if not candidate_usernames:
        ignored_handles = RESERVED_GITHUB_PATHS | {
            "link", "profile", "url", "com", "https", "http", "repo", "repositories",
            "user", "username", "account", "id", "handle", "none", "na", "actions", "pages", "copilot"
        }
        for txt in (plumber_text, raw_text):
            for m in re.finditer(
                r"\bgithub\s*[:|-]\s*@?([A-Za-z0-9](?:[A-Za-z0-9-]{1,37}[A-Za-z0-9]))\b",
                txt,
                re.IGNORECASE,
            ):
                handle = m.group(1).strip("-")
                if handle and handle.lower() not in ignored_handles:
                    candidate_usernames.append(handle)
                    break
            if candidate_usernames:
                break

    if not candidate_usernames:
        return None, None, []

    primary_user = candidate_usernames[0]
    profile_url = f"https://github.com/{primary_user}"
    return profile_url, primary_user, repo_urls


def _extract_name_from_pdf_words(page: pdfplumber.page.Page) -> Optional[str]:
    """Use Page 1 character font sizes to identify the candidate's name."""
    try:
        words = page.extract_words(extra_attrs=["size"])
    except Exception:
        return None
    if not words:
        return None

    top_words = [w for w in words if w.get("top", 999) < 180 and w.get("text", "").strip()]
    if not top_words:
        return None

    # Filter out obvious contact words (emails, urls, phones, symbols) before finding max font size
    candidate_pool = []
    for w in top_words:
        t = w["text"].strip(" |•·—–-:,#§()")
        tl = t.lower()
        if not t or "@" in t or "http" in tl or ".com" in tl or ".in" in tl or ".dev" in tl:
            continue
        if re.search(r"\d{4,}", t):
            continue
        if "(cid:" in tl:
            continue
        candidate_pool.append(w)

    if not candidate_pool:
        return None

    max_size = max(w["size"] for w in candidate_pool)
    largest_words = [w for w in candidate_pool if abs(w["size"] - max_size) < 0.9]
    if not largest_words:
        return None

    first_top = min(w["top"] for w in largest_words)
    # Allow up to 35pt vertical span for 2-line large headers (e.g., Prathamesh / Patil)
    name_tokens: List[str] = []
    for w in largest_words:
        if abs(w["top"] - first_top) > 36:
            continue
        token = w["text"].strip(" |•·—–-:,#§()")
        if not token:
            continue
        if token.lower() in ROLE_STOPWORDS:
            break
        if re.match(r"^[A-Za-z][A-Za-z.\'-]*$", token):
            name_tokens.append(token)
        if len(name_tokens) >= 5:
            break

    if len(name_tokens) == 1:
        first_x0 = largest_words[0].get("x0", 0)
        for w in candidate_pool:
            if w in largest_words:
                continue
            if 5 < (w["top"] - first_top) < 38 and abs(w.get("x0", 0) - first_x0) < 35 and w["size"] >= 15:
                tok = w["text"].strip(" |•·—–-:,#§()")
                if tok and tok.lower() not in ROLE_STOPWORDS and re.match(r"^[A-Za-z][A-Za-z.\'-]*$", tok):
                    name_tokens.append(tok)
                    break

    if name_tokens:
        return " ".join(name_tokens)
    return None


def fallback_extract_name(text: str) -> str:
    """Fallback heuristic to extract candidate name from top lines of text."""
    for raw_line in text.splitlines()[:12]:
        line = re.sub(r"\(cid:\d+\)", " ", raw_line).strip()
        if not line:
            continue
        # Strip inline phone/email/location pipes if present
        first_segment = re.split(r"[|•·—–]|\+\d", line)[0].strip()
        if not first_segment:
            continue
        low = first_segment.lower()
        if "@" in low or "http" in low or ".com" in low or "linkedin" in low or "github" in low:
            continue
        if any(w in low.split() for w in ("summary", "education", "experience", "skills", "objective", "resume", "curriculum")):
            continue
        tokens = [t for t in first_segment.split() if re.match(r"^[A-Za-z][A-Za-z.\'-]*$", t)]
        if 1 <= len(tokens) <= 5:
            return " ".join(tokens)
    return "Unknown Candidate"


def split_into_sections(text: str) -> Dict[str, str]:
    """
    Split resume text into logical sections (summary, skills, experience, projects, education, etc.)
    so downstream filters and scorers can separate skills-list mentions from project/work evidence.
    """
    sections: Dict[str, List[str]] = {"header": []}
    current_section = "header"

    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        clean_header = re.sub(r"[^A-Za-z0-9\s/&()-]", "", line).strip().lower()
        matched_sec: Optional[str] = None
        if len(clean_header) <= 48:
            for sec_name, pattern in SECTION_HEADERS_MAP:
                if re.match(pattern, clean_header, re.IGNORECASE):
                    matched_sec = sec_name
                    break
        if matched_sec:
            current_section = matched_sec
            sections.setdefault(current_section, [])
        else:
            sections.setdefault(current_section, []).append(line)

    return {k: "\n".join(v) for k, v in sections.items() if v}


def extract_matched_skills(text: str) -> List[str]:
    """Extract canonical matched skills from resume text."""
    matched: List[str] = []
    for display_name, pattern, _cat in SKILL_PATTERNS:
        if re.search(pattern, text, re.IGNORECASE):
            if display_name not in matched:
                matched.append(display_name)
    return matched


def _parse_pdf(file_path: Path) -> Tuple[str, str, List[str], Optional[str]]:
    """Extract layout-aware text, line-based text, hyperlinks, and font-based name from a PDF."""
    hyperlinks: List[str] = []
    plumber_pages_text: List[str] = []
    font_name: Optional[str] = None

    with pdfplumber.open(file_path) as pdf:
        if not pdf.pages:
            raise ValueError("PDF file contains 0 pages.")
        font_name = _extract_name_from_pdf_words(pdf.pages[0])
        for page in pdf.pages:
            page_txt = page.extract_text() or ""
            if page_txt:
                plumber_pages_text.append(page_txt)
            for link in page.hyperlinks or []:
                uri = link.get("uri")
                if uri and isinstance(uri, str):
                    clean_uri = uri.strip()
                    if clean_uri and clean_uri not in hyperlinks:
                        hyperlinks.append(clean_uri)

    plumber_text = "\n".join(plumber_pages_text)

    # Also run pdfminer LAParams extraction for clean multi-column grouping
    try:
        miner_text = pdfminer_extract_text(str(file_path)) or ""
    except Exception:
        miner_text = ""

    # Secondary raw PDF byte scan for /URI annotations if hyperlinks list is empty
    if not hyperlinks:
        try:
            raw_bytes = file_path.read_bytes()
            for m in re.finditer(rb"/URI\s*\(([^)]+)\)", raw_bytes):
                uri_str = m.group(1).decode("utf-8", errors="ignore").strip()
                if uri_str and uri_str not in hyperlinks:
                    hyperlinks.append(uri_str)
        except Exception:
            pass

    miner_lines = [l.strip() for l in miner_text.splitlines() if l.strip()]
    # If pdfminer collapsed lines due to missing FontBBox (e.g., candidate_20.pdf), use plumber_text
    if len(miner_lines) < 5 and len(plumber_text) > 100:
        primary_text = plumber_text
    else:
        primary_text = miner_text if len(miner_text.strip()) >= len(plumber_text.strip()) * 0.7 else plumber_text

    return primary_text, plumber_text, hyperlinks, font_name


def _parse_docx(file_path: Path) -> Tuple[str, List[str]]:
    """Extract text and hyperlinks from a .docx file (bonus feature)."""
    paragraphs: List[str] = []
    hyperlinks: List[str] = []
    with zipfile.ZipFile(file_path) as zf:
        if "word/document.xml" not in zf.namelist():
            raise ValueError("Invalid DOCX: missing word/document.xml")
        xml_bytes = zf.read("word/document.xml")
        tree = ET.fromstring(xml_bytes)
        for p in tree.iter("{http://schemas.openxmlformats.org/wordprocessingml/2006/main}p"):
            texts = [
                t.text
                for t in p.iter("{http://schemas.openxmlformats.org/wordprocessingml/2006/main}t")
                if t.text
            ]
            if texts:
                paragraphs.append("".join(texts))

        if "word/_rels/document.xml.rels" in zf.namelist():
            rels_bytes = zf.read("word/_rels/document.xml.rels")
            rels_tree = ET.fromstring(rels_bytes)
            for rel in rels_tree.iter():
                target = rel.attrib.get("Target")
                if target and ("http://" in target or "https://" in target or "mailto:" in target):
                    if target not in hyperlinks:
                        hyperlinks.append(target)

    return "\n".join(paragraphs), hyperlinks


def parse_resume_file(file_path: Path) -> ParsedResume:
    """
    Fault-tolerant entry point to parse a single resume file (.pdf, .docx, or .txt).
    Never raises an unhandled exception; populates parse_error on failure.
    """
    file_name = file_path.name
    try:
        if not file_path.exists() or file_path.stat().st_size == 0:
            return ParsedResume(
                file_name=file_name,
                file_path=str(file_path),
                parse_error="File is empty or does not exist.",
            )

        content_hash = compute_file_hash(file_path)
        ext = file_path.suffix.lower()

        if ext == ".pdf":
            raw_text, plumber_text, hyperlinks, font_name = _parse_pdf(file_path)
        elif ext == ".docx":
            raw_text, hyperlinks = _parse_docx(file_path)
            plumber_text = raw_text
            font_name = None
        elif ext == ".txt":
            raw_text = file_path.read_text(encoding="utf-8", errors="replace")
            plumber_text = raw_text
            hyperlinks = []
            font_name = None
        else:
            return ParsedResume(
                file_name=file_name,
                file_path=str(file_path),
                content_hash=content_hash,
                parse_error=f"Unsupported file extension: {ext}",
            )

        combined_text_check = (raw_text + "\n" + plumber_text).strip()
        if len(combined_text_check) < 20:
            return ParsedResume(
                file_name=file_name,
                file_path=str(file_path),
                content_hash=content_hash,
                parse_error="Extracted resume text is empty or unreadable (possible scanned image without OCR).",
            )

        candidate_name = font_name or fallback_extract_name(raw_text)
        if candidate_name == "Unknown Candidate" and plumber_text:
            candidate_name = fallback_extract_name(plumber_text)

        email = extract_email(combined_text_check, hyperlinks)
        phone = extract_phone(combined_text_check)
        github_url, github_username, github_repo_urls = extract_github_info(
            raw_text, plumber_text, hyperlinks
        )
        sections = split_into_sections(raw_text)
        matched_skills = extract_matched_skills(combined_text_check)

        return ParsedResume(
            file_name=file_name,
            file_path=str(file_path),
            content_hash=content_hash,
            candidate_name=candidate_name,
            email=email,
            phone=phone,
            github_url=github_url,
            github_username=github_username,
            github_repo_urls=github_repo_urls,
            raw_text=raw_text,
            plumber_text=plumber_text,
            hyperlinks=hyperlinks,
            sections=sections,
            matched_skills=matched_skills,
        )
    except Exception as exc:
        return ParsedResume(
            file_name=file_name,
            file_path=str(file_path),
            parse_error=f"Failed to parse resume: {type(exc).__name__}: {exc}",
        )
