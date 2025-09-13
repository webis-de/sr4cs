"""
DuckDuckGo search for paper titles (strict matching, no retries, no PDFs)
- Extract abstracts from HTML (heading-first, domain-aware, JSON-LD aware)
- Uses Selenium + ChromeDriver
- Logs to file only (no stdout)
- Saves progress every N titles (configurable)
- Respects robots.txt (no aggressive crawling)
- Avoids known aggregators and non-scientific domains
- Skips denied domains (hard skip)
- Rejects low-quality abstracts (boilerplate, too short, non-scientific)
- Optional: fetch and parse PDFs via GROBID (configurable)
"""

import os, re, json, time, random, logging, math, ast
from datetime import datetime
from typing import Dict, Optional, Tuple, List, Any, Iterable

import pandas as pd
from bs4 import BeautifulSoup, Tag
from fuzzywuzzy import fuzz
from tqdm import tqdm

from langdetect import detect, DetectorFactory

DetectorFactory.seed = 0

from selenium import webdriver
from selenium.webdriver import ChromeOptions
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.common.action_chains import ActionChains
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import (
    TimeoutException,
    WebDriverException,
    StaleElementReferenceException,
)

import requests

# =========================
# CONFIG
# =========================
INPUT_PARQUET = "../../../data/filled/combined_refs_filtered.parquet"
OUTPUT_PARQUET = None
TITLE_COL = "title_norm"
SAVE_EVERY = 20
DELAY_SEC = 2.0

MAX_RETRIES = 1

LOG_FILE = "../../../logs/ddg_search_combined.log"

# Matching parameters (SERP result-title vs query title)
MAX_HITS = 10
CONF_THRESH = 0.90
CONF_STRONG = 0.95
SHORT_TITLE_K = 5

# Selenium / crawling hygiene
HEADLESS = True
SEARCH_WAIT = 12
NAV_WAIT = 25
CLICK_PAUSE = (0.8, 1.6)
DDG_SLEEP = (0.6, 1.2)
BETWEEN_TITLES = (2.0, 4.0)
SCROLL_STEPS = (2, 4)
PER_DOMAIN_COOLDOWN_SEC = 8

# Abstract acceptance thresholds (strict)
MIN_ABS_CHARS = 200
MIN_ABS_WORDS = 60
MIN_SENTENCES = 2
MAX_URL_QUERYLEN = 300
MAX_ABS_CHARS_FINAL = 6000  # allow longer abstracts
MAX_ABS_PARA_JOIN = 6

# PDF policy
REJECT_PDFS = True
USE_GROBID_FOR_PDFS = False
GROBID_URL = "http://localhost:8070"
GROBID_TIMEOUT = 60
PDF_FETCH_TIMEOUT = 45

# Denylist (hard-skip on SERP and after redirects)
DOMAIN_DENYLIST = {
    "youtube.com",
    "youtu.be",
    "amazon.com",
    "amazon.de",
    "smile.amazon.com",
    "m.youtube.com",
}

# ResearchGate handling: "clean_description" | "strict_meta"
RESEARCHGATE_POLICY = "clean_description"

# Optional rotating proxies
PROXIES: List[str] = [
    # "http://user:pass@res1.example:port",
]

UA = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit(537.36) (KHTML, like Gecko) "
    "Chrome/127.0.0.0 Safari/537.36"
)

# Domains we like
PREFERRED_DOMAINS = [
    "pubmed.ncbi.nlm.nih.gov",
    "pmc.ncbi.nlm.nih.gov",
    "onlinelibrary.wiley.com",
    "link.springer.com",
    "nature.com",
    "sciencedirect.com",
    "ieeexplore.ieee.org",
    "dl.acm.org",
    "arxiv.org",
    "tandfonline.com",
    "oup.com",
    "sagepub.com",
    "jstor.org",
]

AGGREGATORS_DOWNRANK = [
    "researchgate.net",
    "semanticscholar.org",
    "proquest.com",
    "academia.edu",
]

# Hosts where we enforce stricter selection (no meta desc / no generic CSS)
STRICT_HOSTS = {"link.springer.com", "sciencedirect.com"}

# =========================
# Logging (FILE ONLY)
# =========================
os.makedirs(os.path.dirname(LOG_FILE) if LOG_FILE else "./logs", exist_ok=True)
LOG_FILE = LOG_FILE or os.path.join(
    "./logs", f"ddg_title_strict_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[logging.FileHandler(LOG_FILE, mode="a", encoding="utf-8")],
)
log = logging.getLogger("ddg_strict")

# =========================
# Utils
# =========================
STOP = {
    "the",
    "a",
    "an",
    "of",
    "in",
    "on",
    "at",
    "to",
    "for",
    "with",
    "by",
    "from",
    "and",
    "or",
    "is",
    "are",
    "be",
    "as",
}
SCI_KEYWORDS = {
    "we propose",
    "we present",
    "we introduce",
    "we show",
    "this paper",
    "this study",
    "results show",
    "methods",
    "methodology",
    "experimental",
    "dataset",
    "model",
    "algorithm",
    "evaluation",
    "performance",
    "conclusion",
    "conclusions",
    "in this paper",
    "our approach",
    "our method",
    "we evaluate",
    "we investigate",
    "findings",
    "we demonstrate",
    "accuracy",
    "precision",
    "recall",
    "baseline",
    "experiments",
    "robustness",
}
BOILERPLATE_PATTERNS = [
    r"all content on this site",
    r"copyright\s*©",
    r"springer nature remains neutral",
    r"terms and conditions",
    r"cookies? (policy|notice)",
    r"by using this site you agree",
    r"rights reserved",
    r"learn how (nature research intelligence|nri) gives you",
    r"this site uses cookies",
    r"publisher (policy|licen[cs]e)",
    r"open access funding provided",
    r"accept (all )?cookies",
    r"sciencedirect is a registered trademark",
    r"elsevier (b\.v\.|inc\.)",
    r"sign in|subscribe now|purchase access",
    r"download pdf|view html|export citation",
    r"these cookies allow us to count visits and traffic sources",
    r"performance of our site",
    r"manage your cookie preferences",
]
BOILERPLATE_RE = re.compile("|".join(BOILERPLATE_PATTERNS), re.I)

COOKIE_ANCESTRY_RE = re.compile(
    r"(cookie|consent|gdpr|onetrust|ot-sdk|privacy|banner|tracking|preferences|cmp|quantcast|trustarc)",
    re.I,
)

ABSTRACT_HEADING_RE = re.compile(
    r"^\s*(abstract|summary|zusammenfassung|résumé|resumen|abstrakt|sommario|samenvatting)\s*:?\s*$",
    re.I,
)

SECTION_STOP_RE = re.compile(
    r"^\s*(highlights?|keywords?|introduction|background|1\.\s|related work|methods?|materials|results?|discussion|conclusion)s?\s*:?\s*$",
    re.I,
)


def _clean(s: str) -> str:
    s = re.sub(r"\s+", " ", str(s).strip())
    s = re.sub(r"[^\w\s]", "", s.lower()).strip()
    return s


def _tokens(s: str) -> List[str]:
    return [t for t in _clean(s).split() if t and t not in STOP]


def title_conf(q: str, cand: str) -> float:
    q2, c2 = _clean(q), _clean(cand)
    if not q2 or not c2:
        return 0.0
    return fuzz.token_set_ratio(q2, c2) / 100.0


def randsleep(a: float, b: float):
    time.sleep(random.uniform(a, b))


def host_from_url(url: str) -> str:
    try:
        return re.search(r"https?://([^/]+)/?", url).group(1).lower()
    except Exception:
        return ""


def norm_host(h: str) -> str:
    h = (h or "").lower()
    return h[4:] if h.startswith("www.") else h


def is_denied_host(url: str) -> bool:
    h = norm_host(host_from_url(url))
    if h in DOMAIN_DENYLIST:
        return True
    return False


def sent_split(s: str) -> List[str]:
    parts = re.split(r"(?<=[.!?])\s+", s.strip())
    return [p for p in parts if len(p) > 0]


def stopword_ratio(text: str) -> float:
    toks = re.findall(r"[A-Za-z]+", text.lower())
    if not toks:
        return 1.0
    stop = sum(1 for t in toks if t in STOP)
    return stop / max(1, len(toks))


def punct_ratio(text: str) -> float:
    if not text:
        return 1.0
    p = sum(1 for c in text if c in ",.;:()[]{}")
    return p / max(1, len(text))


def has_science_keywords(text: str) -> bool:
    low = text.lower()
    return any(k in low for k in SCI_KEYWORDS)


def looks_like_boilerplate(text: str) -> bool:
    t = text.strip()
    if len(t) < MIN_ABS_CHARS:
        return True
    if BOILERPLATE_RE.search(t):
        return True
    if stopword_ratio(t) > 0.65:
        return True
    if len(sent_split(t)) < MIN_SENTENCES:
        return True
    if punct_ratio(t) < 0.005:
        return True
    if len(t.split()) >= MIN_ABS_WORDS and not has_science_keywords(t):
        if len(t) < 900:
            return True
    try:
        lang = detect(t[:1000])
        if lang not in ("en", "de", "fr", "es"):
            return True
    except Exception:
        pass
    return False


def is_inside_blacklisted(el: Optional[Tag]) -> bool:
    try:
        cur = el
        depth = 0
        while cur and isinstance(cur, Tag) and depth < 8:
            id_ = cur.get("id") or ""
            cls = " ".join(cur.get("class") or [])
            if COOKIE_ANCESTRY_RE.search(id_) or COOKIE_ANCESTRY_RE.search(cls):
                return True
            cur = cur.parent
            depth += 1
    except Exception:
        return False
    return False


# =========================
# Author helpers (query boost)
# =========================
def first_author_family(cell: Any) -> Optional[str]:
    if cell is None or (isinstance(cell, float) and math.isnan(cell)):
        return None
    try:
        obj = cell
        if isinstance(cell, str):
            obj = ast.literal_eval(cell)
        if isinstance(obj, list) and obj:
            item = obj[0]
            if isinstance(item, dict):
                fam = item.get("family")
                if isinstance(fam, str) and len(fam) >= 2:
                    return fam.strip()
    except Exception:
        return None
    return None


# =========================
# PDF helpers (GROBID)
# =========================
def is_pdf_url(url: str) -> bool:
    return url.lower().endswith(".pdf")


def fetch_pdf_bytes(url: str, referer: Optional[str] = None) -> Optional[bytes]:
    try:
        headers = {"User-Agent": UA}
        if referer:
            headers["Referer"] = referer
        r = requests.get(url, headers=headers, timeout=PDF_FETCH_TIMEOUT)
        if (
            r.status_code == 200
            and r.content
            and (
                r.headers.get("Content-Type", "").lower().startswith("application/pdf")
                or is_pdf_url(url)
            )
        ):
            return r.content
        return None
    except Exception as e:
        log.info(f"PDF fetch error: {e}")
        return None


def grobid_fulltext_tei_from_pdf(pdf_bytes: bytes) -> Optional[str]:
    try:
        files = {"input": ("doc.pdf", pdf_bytes, "application/pdf")}
        params = {
            "consolidateHeader": "1",
            "includeRawCitations": "0",
            "teiCoordinates": "0",
        }
        url = f"{GROBID_URL.rstrip('/')}/api/processFulltextDocument"
        r = requests.post(url, files=files, data=params, timeout=GROBID_TIMEOUT)
        if r.status_code == 200 and r.text:
            return r.text
        log.info(f"GROBID error: status={r.status_code}")
        return None
    except Exception as e:
        log.info(f"GROBID request failed: {e}")
        return None


def extract_abstract_from_tei(tei_xml: str) -> Optional[str]:
    try:
        soup = BeautifulSoup(tei_xml, "xml")
        abst = soup.find("abstract")
        if abst:
            parts = [
                p.get_text(" ", strip=True)
                for p in abst.find_all(["p", "div"], recursive=True)
            ]
            txt = " ".join([t for t in parts if t]).strip()
            if txt:
                return txt
        div = soup.find("div", {"type": "abstract"})
        if div:
            txt = div.get_text(" ", strip=True)
            if txt:
                return txt
    except Exception as e:
        log.info(f"TEI parse error: {e}")
    return None


# =========================
# Robust soup
# =========================
def robust_soup(html: str) -> BeautifulSoup:
    try:
        return BeautifulSoup(html, "lxml")
    except Exception:
        pass
    try:
        return BeautifulSoup(html, "html.parser")
    except Exception:
        pass
    try:
        return BeautifulSoup(html, "html5lib")  # requires html5lib installed
    except Exception:
        cleaned = re.sub(r"\s+xmlns(:\w+)?=\{[^}]*\}", "", html)
        return BeautifulSoup(cleaned, "html.parser")


# =========================
# Selenium setup
# =========================
def make_options(proxy: Optional[str] = None) -> ChromeOptions:
    opts = ChromeOptions()
    if HEADLESS:
        opts.add_argument("--headless=new")
    opts.add_argument("--disable-gpu")
    opts.add_argument("--no-sandbox")
    opts.add_argument("--disable-dev-shm-usage")
    opts.add_argument("--window-size=1280,1200")
    opts.add_argument("--lang=en-US,en;q=0.9")
    opts.add_argument(f"--user-agent={UA}")
    # Avoid waiting for all subresources; helps with stalls
    opts.page_load_strategy = "eager"
    opts.add_experimental_option("excludeSwitches", ["enable-automation"])
    opts.add_experimental_option("useAutomationExtension", False)
    if proxy:
        opts.add_argument(f"--proxy-server={proxy}")
    return opts


def new_driver(proxy: Optional[str] = None):
    driver = webdriver.Chrome(options=make_options(proxy))
    driver.set_page_load_timeout(NAV_WAIT)
    try:
        driver.set_script_timeout(15)
    except Exception:
        pass
    try:
        driver.execute_cdp_cmd(
            "Page.addScriptToEvaluateOnNewDocument",
            {
                "source": "Object.defineProperty(navigator, 'webdriver', {get: () => undefined})"
            },
        )
    except Exception:
        pass
    return driver


# =========================
# DuckDuckGo search
# =========================
def ddg_collect_cards(driver, query: str) -> List[Tuple[str, str, Any]]:
    driver.get("https://duckduckgo.com/")
    WebDriverWait(driver, 12).until(
        EC.presence_of_element_located(
            (By.CSS_SELECTOR, 'input[id="searchbox_input"], input[name="q"]')
        )
    )
    box = driver.find_element(
        By.CSS_SELECTOR, 'input[id="searchbox_input"], input[name="q"]'
    )
    box.clear()
    box.send_keys(query)
    randsleep(0.1, 0.3)
    box.send_keys(Keys.ENTER)

    WebDriverWait(driver, SEARCH_WAIT).until(
        EC.presence_of_element_located((By.CSS_SELECTOR, "a"))
    )
    randsleep(*DDG_SLEEP)

    cards = []
    anchors = driver.find_elements(By.CSS_SELECTOR, 'a[data-testid="result-title-a"]')
    if not anchors:
        anchors = driver.find_elements(By.CSS_SELECTOR, "a.result__a")
    for a in anchors[:MAX_HITS]:
        try:
            href = a.get_attribute("href") or ""
            if len(href) > MAX_URL_QUERYLEN:
                continue
            rtitle = a.text
            if href and rtitle:
                cards.append((rtitle, href, a))
        except StaleElementReferenceException:
            continue
    return cards


def non_pdf_bias(url: str) -> int:
    score = 0
    host = host_from_url(url)
    if not is_pdf_url(url):
        score += 2
    if any(host.endswith(d) or host == d for d in PREFERRED_DOMAINS):
        score += 2
    if any(d in host for d in AGGREGATORS_DOWNRANK):
        score -= 1
    return score


# =========================
# Extraction helpers (heading-first & domain-aware)
# =========================
def trim_abstract(txt: str, src: str) -> str:
    # Keep full text for jsonld/citation_abstract/domain selectors (they're already scoped)
    if (
        src.startswith("jsonld:")
        or src.startswith("meta_citation_abstract")
        or src.startswith("domain:")
    ):
        return txt[:MAX_ABS_CHARS_FINAL].strip()
    paras = re.split(r"\n{2,}|(?<=\.)\s{2,}", txt.strip())
    if len(paras) <= 1:
        paras = [p.strip() for p in re.split(r"(?<=\.)\s+", txt.strip())]
    kept = []
    for p in paras:
        if not p:
            continue
        kept.append(p)
        if len(" ".join(kept)) >= MAX_ABS_CHARS_FINAL or len(kept) >= MAX_ABS_PARA_JOIN:
            break
    final = " ".join(kept).strip()
    return final if final else txt[:MAX_ABS_CHARS_FINAL].strip()


def text_from_nodes(nodes: List[Tag]) -> str:
    parts = []
    for n in nodes:
        if isinstance(n, Tag):
            t = n.get_text(" ", strip=True)
            if t and not BOILERPLATE_RE.search(t) and not is_inside_blacklisted(n):
                parts.append(t)
    return " ".join(parts).strip()


def gather_under_heading(h: Tag) -> str:
    collected: List[str] = []
    paras = 0
    node = h.find_next_sibling()
    while node and paras < MAX_ABS_PARA_JOIN:
        if isinstance(node, Tag):
            if node.name in ["h1", "h2", "h3", "h4", "h5", "h6"]:
                break
            if node.name in ["p", "div", "section"]:
                if SECTION_STOP_RE.match(node.get_text(" ", strip=True) or ""):
                    break
                if node.name == "p":
                    if not is_inside_blacklisted(node):
                        txt = node.get_text(" ", strip=True)
                        if txt:
                            collected.append(txt)
                            paras += 1
                else:
                    for p in node.find_all("p", recursive=False):
                        if paras >= MAX_ABS_PARA_JOIN:
                            break
                        if not is_inside_blacklisted(p):
                            t = p.get_text(" ", strip=True)
                            if t:
                                collected.append(t)
                                paras += 1
        node = node.find_next_sibling()
    return " ".join(collected).strip()


def extract_by_heading(soup: BeautifulSoup) -> Optional[Tuple[str, str]]:
    candidates: List[Tuple[str, str]] = []
    for h in soup.find_all(re.compile(r"^h[1-6]$")):
        title = (h.get_text(" ", strip=True) or "").strip()
        if ABSTRACT_HEADING_RE.match(title):
            txt = gather_under_heading(h)
            if txt:
                candidates.append((txt, f"heading:{title.lower()}"))

    # DL pattern (dt: Abstract, dd: text)
    for dt in soup.find_all("dt"):
        if ABSTRACT_HEADING_RE.match(dt.get_text(" ", strip=True) or ""):
            dd = dt.find_next_sibling("dd")
            if dd:
                paras = [p.get_text(" ", strip=True) for p in dd.find_all("p")] or [
                    dd.get_text(" ", strip=True)
                ]
                txt = " ".join([p for p in paras if p]).strip()
                if txt:
                    candidates.append((txt, "dl:abstract"))

    # Pick best by length + presence of sci terms
    best_txt, best_src, best_score = None, "", -1
    for txt, src in candidates:
        if looks_like_boilerplate(txt):
            continue
        sc = (min(len(txt), 1600) / 1600.0) + (
            0.3 if has_science_keywords(txt) else 0.0
        )
        if sc > best_score:
            best_txt, best_src, best_score = txt, src, sc
    if best_txt:
        return best_txt, best_src
    return None


# ProQuest: Headnote → Summary
def extract_proquest_summary(soup: BeautifulSoup) -> Optional[Tuple[str, str]]:
    hn = soup.select_one(".Headnote_content")
    if not hn:
        return None
    ps = hn.find_all("p")
    if not ps:
        return None
    txts = []
    seen_summary = False
    for p in ps:
        t = p.get_text(" ", strip=True)
        if not t:
            continue
        if not seen_summary:
            if re.fullmatch(r"\s*summary\s*", t, flags=re.I):
                seen_summary = True
            continue
        if SECTION_STOP_RE.match(t):
            break
        txts.append(t)
        if len(" ".join(txts)) > MAX_ABS_CHARS_FINAL:
            break
    final = " ".join(txts).strip()
    if final:
        return final, "proquest:headnote_summary"
    return None


# ResearchGate: clean meta description to get abstract-ish summary
def extract_researchgate_clean_description(
    soup: BeautifulSoup,
) -> Optional[Tuple[str, str]]:
    m = soup.find("meta", {"name": "description"})
    content = m.get("content") if m else None
    if not content:
        og = soup.find("meta", {"property": "og:description"})
        content = og.get("content") if og else None
    if not content:
        return None
    txt = content.strip()
    txt = re.sub(r"^\s*request\s*pdf\s*\|\s*", "", txt, flags=re.I)
    txt = re.sub(r"\|\s*find,\s*read.*?researchgate\s*$", "", txt, flags=re.I)
    if "|" in txt:
        parts = [p.strip() for p in txt.split("|") if p.strip()]
        if len(parts) >= 2:
            parts_sorted = sorted(parts, key=len, reverse=True)
            txt = parts_sorted[0]
    txt = re.sub(r"\s+", " ", txt).strip()
    if len(txt) < MIN_ABS_CHARS // 2:
        return None
    if BOILERPLATE_RE.search(txt):
        return None
    return txt, "researchgate:meta_description_clean"


DOMAIN_SELECTORS = {
    "pmc.ncbi.nlm.nih.gov": [
        "#abstract",
        "section.abstract",
        "div.abstr",
        ".tsec .abstract",
    ],
    "pubmed.ncbi.nlm.nih.gov": ["section#abstract", "div.abstr"],
    "onlinelibrary.wiley.com": ["section#abstract", ".article-section__content"],
    # Springer: Abs1-based + content section
    "link.springer.com": [
        ".c-article-section__content",
        "section#Abs1",
        "#Abs1-content",
    ],
    "nature.com": [".c-article-section__content"],
    # ScienceDirect: author abstract ONLY (never highlights/graphical)
    "sciencedirect.com": [
        "div.Abstracts div.abstract.author",
        "div.Abstracts div.abstract.author p",
        "div#abstracts div.abstract.author",
        "meta[name='citation_abstract']",  # meta, but strong prior when present
    ],
    "dl.acm.org": [".abstractSection", ".abstractInFull"],
    "ieeexplore.ieee.org": [
        ".abstract-text",
        "#abstract-text",
        ".stats-document-abstract",
    ],
    "tandfonline.com": ["section.abstract", "div.abstractSection"],
    "arxiv.org": [".abstract", "blockquote.abstract"],
}

GENERIC_SELECTORS = [
    # generic abstract-ish hooks (disabled for STRICT_HOSTS below)
    "#abstract",
    ".abstract",
    ".abstract-text",
    ".abstractSection",
    "[data-testid='abstract']",
    "[role='doc-abstract']",
    "[data-type='abstract']",
    ".c-article-section__content",
    ".chapter-abstract",
    ".hlFld-Abstract",
    ".article-section__content",
    "#Abs1",
    "#Abs1-content",
    "section.abstract",
    ".abstract-content",
    ".simple-para",
    "section#abstract p",
]

META_NAMES = [
    ("name", "citation_abstract"),
    ("name", "description"),
    ("property", "og:description"),
    ("name", "DC.Description"),
]


def _iter_json_objects(data: Any) -> Iterable[dict]:
    """Yield all dict objects from a JSON structure."""
    if isinstance(data, dict):
        yield data
        for v in data.values():
            yield from _iter_json_objects(v)
    elif isinstance(data, list):
        for item in data:
            yield from _iter_json_objects(item)


def _get_first_str(o: dict, keys: List[str]) -> Optional[str]:
    for k in keys:
        v = o.get(k)
        if isinstance(v, str) and v.strip():
            return v.strip()
        if isinstance(v, dict):
            vv = v.get("@value")
            if isinstance(vv, str) and vv.strip():
                return vv.strip()
    return None


def extract_jsonld_deep(soup: BeautifulSoup, host: str) -> List[Tuple[str, str, None]]:
    """Prefer ScholarlyArticle abstract/description; recurse through mainEntity and nested JSON-LD."""
    cands: List[Tuple[str, str, None]] = []
    for script in soup.find_all("script", {"type": "application/ld+json"}):
        raw = script.string or ""
        if not raw.strip():
            continue
        try:
            data = json.loads(raw)
        except Exception:
            try:
                fixed = raw.replace("\u0000", "").strip()
                data = json.loads(fixed)
            except Exception:
                continue

        for obj in _iter_json_objects(data):
            types = obj.get("@type")
            if isinstance(types, list):
                types_low = [str(t).lower() for t in types]
            else:
                types_low = [str(types).lower()] if types else []

            if "webpage" in types_low and isinstance(
                obj.get("mainEntity"), (dict, list)
            ):
                pass

            if any(t in ("scholarlyarticle", "article") for t in types_low):
                v = _get_first_str(obj, ["abstract", "description"])
                if v:
                    cands.append((v.strip(), "jsonld:scholarly", None))
            else:
                v = _get_first_str(obj, ["abstract", "description"])
                if v and len(v) > 200:
                    cands.append((v.strip(), "jsonld:description", None))
    return cands


def extract_candidates(
    soup: BeautifulSoup, host: str
) -> List[Tuple[str, str, Optional[Tag]]]:
    host_norm = norm_host(host)
    cands: List[Tuple[str, str, Optional[Tag]]] = []

    # 0) JSON-LD deep (preferred for Springer/Elsevier)
    for txt, src, _ in extract_jsonld_deep(soup, host_norm):
        cands.append((txt, src, None))

    # 1) Section-heading (Abstract/Summary/…)
    res = extract_by_heading(soup)
    if res:
        cands.append((res[0], res[1], None))

    # 1b) ProQuest Summary
    if "proquest.com" in host_norm:
        pq = extract_proquest_summary(soup)
        if pq:
            cands.append((pq[0], pq[1], None))

    # 1c) ResearchGate cleaned description
    if "researchgate.net" in host_norm and RESEARCHGATE_POLICY == "clean_description":
        rg = extract_researchgate_clean_description(soup)
        if rg:
            cands.append((rg[0], rg[1], None))

    # 2) Domain-aware CSS (very strong priors)
    for sel in DOMAIN_SELECTORS.get(host_norm, []):
        el = soup.select_one(sel)
        if el and not is_inside_blacklisted(el):
            if host_norm == "sciencedirect.com":
                if (
                    isinstance(el, Tag)
                    and el.name == "meta"
                    and el.get("name") == "citation_abstract"
                ):
                    txt = el.get("content", "").strip()
                else:
                    classes = (
                        " ".join(el.get("class", [])) if hasattr(el, "get") else ""
                    )
                    if (
                        "abstract" in classes
                        and "author" not in classes
                        and el.name != "meta"
                    ):
                        # skip highlights/graphical/topic abstracts
                        continue
                    txt = el.get_text(" ", strip=True)
            else:
                txt = (
                    el.get_text(" ", strip=True)
                    if el.name != "meta"
                    else (el.get("content", "").strip())
                )

            if txt:
                src = f"domain:{host_norm}:{sel}"
                cands.append((txt, src, el))

    # 3) Meta tags (block meta description on strict hosts)
    for k, v in META_NAMES:
        if host_norm in STRICT_HOSTS and v in ("description", "og:description"):
            continue
        m = soup.find("meta", {k: v})
        if m and m.get("content"):
            src = f"meta_{v}"
            cands.append((m["content"].strip(), src, None))

    # ResearchGate strict policy: only accept citation_abstract; drop raw meta
    if "researchgate.net" in host_norm and RESEARCHGATE_POLICY == "strict_meta":
        keep = []
        for t, s, el in cands:
            if s.startswith("meta_citation_abstract") or not (s.startswith("meta_")):
                keep.append((t, s, el))
        cands = keep

    # 5) Generic selectors (skip for strict hosts)
    if host_norm not in STRICT_HOSTS:
        for sel in GENERIC_SELECTORS:
            el = soup.select_one(sel)
            if el and not is_inside_blacklisted(el):
                txt = el.get_text(" ", strip=True)
                if txt:
                    cands.append((txt, f"css:{sel}", el))

    # Deduplicate
    seen = set()
    uniq = []
    for t, s, el in cands:
        key = (t[:200], s)
        if key in seen:
            continue
        seen.add(key)
        uniq.append((t, s, el))
    return uniq


def score_candidate(text: str, host: str, source: str) -> Tuple[float, List[str]]:
    reasons = []
    host_norm = norm_host(host)
    if BOILERPLATE_RE.search(text):
        reasons.append("boilerplate_regex")
    words = len(text.split())
    if len(text) < MIN_ABS_CHARS or words < MIN_ABS_WORDS:
        reasons.append(f"too_short:{len(text)}c/{words}w")
    nsent = len(sent_split(text))
    if nsent < MIN_SENTENCES:
        reasons.append(f"few_sentences:{nsent}")
    if stopword_ratio(text) > 0.65:
        reasons.append("high_stopword_ratio")
    if punct_ratio(text) < 0.005:
        reasons.append("low_punct_ratio")

    has_sci = has_science_keywords(text)
    if not has_sci and len(text) < 900:
        if not (
            source.startswith("meta_citation_abstract")
            or source.startswith("heading:")
            or source.startswith("proquest:")
            or source.startswith("researchgate:")
            or source.startswith("jsonld:")
            or source.startswith("domain:")
        ):
            reasons.append("no_science_kw")

    if host_norm in STRICT_HOSTS and (
        source.startswith("meta_description")
        or source.startswith("meta_og:description")
    ):
        reasons.append("meta_desc_blocked_on_strict_host")

    try:
        lang = detect(text[:1000])
        if lang not in ("en", "de", "fr", "es"):
            reasons.append(f"lang_{lang}")
    except Exception:
        pass
    if reasons:
        return 0.0, reasons

    score = 0.0
    score += min(1.0, len(text) / 1600.0) * 0.45
    score += 0.25 if has_sci else 0.0
    score += max(0.0, (punct_ratio(text) - 0.01)) * 5
    score += max(0.0, (0.65 - stopword_ratio(text)))
    score += max(0.0, (nsent - 2)) * 0.05

    # Strong source priors
    if source.startswith("domain:"):
        score += 0.9
    if source.startswith("jsonld:scholarly"):
        score += 0.9
    if source.startswith("jsonld:description"):
        score += 0.7
    if source.startswith("heading:"):
        score += 0.7
    if source.startswith("meta_citation_abstract"):
        score += 0.6
    if source.startswith("proquest:"):
        score += 0.6
    if source.startswith("researchgate:"):
        score += 0.45
    if source.startswith("css:.abstract"):
        score += 0.15
    if source.startswith("meta_description") or source.startswith(
        "meta_og:description"
    ):
        score -= 0.4
    return score, []


def strict_select_abstract(soup: BeautifulSoup, host: str) -> Tuple[Optional[str], str]:
    cands = extract_candidates(soup, host)
    if not cands:
        return None, "no_candidates"

    best_txt, best_src, best_score = None, "", 0.0
    for i, (txt, src, el) in enumerate(cands, 1):
        if el is not None and is_inside_blacklisted(el):
            log.info(f"   ✖ reject [{i}] {src} — reasons: cookie_ancestry")
            continue
        score, reasons = score_candidate(txt, host, src)
        if score <= 0:
            log.info(f"   ✖ reject [{i}] {src} — reasons: {', '.join(reasons)}")
            continue
        trimmed = trim_abstract(txt, src)
        log.info(f"   ✓ candidate [{i}] {src} — score={score:.2f} len={len(trimmed)}")
        if score > best_score:
            best_score, best_txt, best_src = score, trimmed, src

    if best_txt:
        return best_txt, best_src
    return None, "all_rejected"


def click_anchor_preserving_referer(driver, anchor_we) -> bool:
    try:
        ActionChains(driver).move_to_element(anchor_we).pause(0.1).click(
            anchor_we
        ).perform()
        return True
    except Exception:
        return False


def extract_from_current_page(driver) -> Tuple[Optional[str], str]:
    # SAFE initial wait: avoid execute_script; just ensure <body> exists
    try:
        WebDriverWait(driver, min(NAV_WAIT, 10)).until(
            EC.presence_of_element_located((By.TAG_NAME, "body"))
        )
    except TimeoutException:
        pass

    handle_cookie_banners(driver)
    soft_scroll(driver)

    # Host-aware quick wait for abstract containers
    try:
        host = (driver.execute_script("return location.hostname") or "").lower()
    except Exception:
        host = ""

    try:
        if "sciencedirect.com" in host:
            WebDriverWait(driver, 3).until(
                EC.presence_of_element_located(
                    (
                        By.CSS_SELECTOR,
                        "div.Abstracts div.abstract.author, meta[name='citation_abstract'], script[type='application/ld+json']",
                    )
                )
            )
        elif "link.springer.com" in host:
            WebDriverWait(driver, 3).until(
                EC.presence_of_element_located(
                    (
                        By.CSS_SELECTOR,
                        "section#Abs1, #Abs1-content, .c-article-section__content",
                    )
                )
            )
    except Exception:
        pass

    html = driver.page_source
    soup = robust_soup(html)
    txt, src = strict_select_abstract(soup, host)
    return txt, src


# =========================
# Cookie banners & scroll
# =========================
def handle_cookie_banners(driver):
    selectors = [
        "#onetrust-accept-btn-handler",
        "button[aria-label*='Accept']",
        "button[aria-label*='Consent']",
        "button[mode='primary']",
        "button.cookie-accept, button.accept, button#acceptAll",
        "button#truste-consent-button",
    ]
    for sel in selectors:
        try:
            els = driver.find_elements(By.CSS_SELECTOR, sel)
            if els:
                els[0].click()
                randsleep(0.2, 0.5)
                break
        except Exception:
            pass


def soft_scroll(driver):
    steps = random.randint(*SCROLL_STEPS)
    for _ in range(steps):
        driver.execute_script(
            "window.scrollBy(0, arguments[0]);", random.randint(300, 900)
        )
        randsleep(0.25, 0.6)


# =========================
# Per-title flow
# =========================
def process_pdf_with_grobid(
    url: str, referer: Optional[str]
) -> Tuple[Optional[str], str]:
    if not USE_GROBID_FOR_PDFS:
        return None, "pdf_rejected"
    if not GROBID_URL:
        return None, "grobid_not_configured"
    pdf = fetch_pdf_bytes(url, referer=referer)
    if not pdf:
        return None, "pdf_fetch_failed"
    tei = grobid_fulltext_tei_from_pdf(pdf)
    if not tei:
        return None, "grobid_failed"
    abs_txt = extract_abstract_from_tei(tei)
    if not abs_txt:
        return None, "tei_no_abstract"
    if looks_like_boilerplate(abs_txt):
        return None, "tei_boilerplate"
    return trim_abstract(abs_txt, "tei"), "grobid_fulltext_tei"


def process_title_with_retries(
    query: str, domain_last_hit: Dict[str, float]
) -> Dict[str, Any]:
    start = time.time()
    abstract, url_used, serp_title = None, None, None

    for attempt in range(1, MAX_RETRIES + 1):
        proxy = random.choice(PROXIES) if PROXIES else None
        log.info("-" * 70)
        log.info(f"Row | Title: {query[:140]}")
        if proxy:
            log.info(f"Proxy: {proxy}")
        log.info(f"🔎 DDG query: {query}")

        try:
            driver = new_driver(proxy)
        except WebDriverException as e:
            wait = 1.0
            log.warning(f"Driver error: {e} — abort (MAX_RETRIES=1)")
            time.sleep(wait)
            break

        try:
            cards = ddg_collect_cards(driver, query)
            ranked = []
            log.info(f"📊 SERP hits: {len(cards)}")
            for i, (rt, url, we) in enumerate(cards, 1):
                if is_denied_host(url):
                    log.info(
                        f"  [{i}] {rt[:100]} — SKIP (denylist): {host_from_url(url)}"
                    )
                    continue
                sc = title_conf(query, rt)
                bias = non_pdf_bias(url)
                total = sc + 0.02 * bias
                log.info(
                    f"  [{i}] {rt[:100]} — conf={sc:.2f} bias={bias:+} total≈{total:.2f}"
                )
                ranked.append((total, sc, rt, url, we))
            ranked.sort(key=lambda x: x[0], reverse=True)

            nonpdf = [
                (sc, rt, url, we)
                for _, sc, rt, url, we in ranked
                if not is_pdf_url(url)
            ]
            pdfs = [
                (sc, rt, url, we) for _, sc, rt, url, we in ranked if is_pdf_url(url)
            ]

            if REJECT_PDFS:
                ranked_seq = nonpdf[:3]
            else:
                ranked_seq = nonpdf[:3] + (pdfs[:1] if USE_GROBID_FOR_PDFS else [])

            if not ranked_seq:
                driver.quit()
                log.info("No acceptable SERP hit; abort (MAX_RETRIES=1)")
                break

            tried = 0
            for sc, rt, url, anchor_we in ranked_seq:
                tried += 1
                serp_title = rt

                if is_denied_host(url):
                    log.info(f"Skip click (denylist): {url}")
                    continue

                host = host_from_url(url)
                last = domain_last_hit.get(host, 0.0)
                wait_more = max(0.0, PER_DOMAIN_COOLDOWN_SEC - (time.time() - last))
                if wait_more > 0:
                    log.info(f"Cooldown for {host}: {wait_more:.1f}s")
                    time.sleep(wait_more)

                if is_pdf_url(url):
                    log.info(f"📄 PDF detected: {url}")
                    if USE_GROBID_FOR_PDFS:
                        abs_txt, note = process_pdf_with_grobid(
                            url, referer="https://duckduckgo.com/"
                        )
                        if abs_txt:
                            abstract = abs_txt
                            url_used = url
                            domain_last_hit[host] = time.time()
                            log.info(
                                f"✅ ACCEPTED abstract via {note}, len={len(abstract)}"
                            )
                            break
                        else:
                            log.info(
                                f"❌ PDF via GROBID failed ({note}); trying next SERP hit"
                            )
                            continue
                    else:
                        log.info("PDF rejected by policy; skipping")
                        continue

                log.info(f"🖱️ Click SERP: {serp_title[:120]} → {url}")
                ok = click_anchor_preserving_referer(driver, anchor_we)
                if not ok:
                    log.info("Anchor click failed; fallback driver.get(url)")
                    driver.get(url)

                randsleep(*CLICK_PAUSE)
                abs_txt, note = extract_from_current_page(driver)
                cur_url = ""
                try:
                    cur_url = driver.current_url
                except Exception:
                    cur_url = url

                if is_denied_host(cur_url):
                    log.info(f"Landing page denylisted → skipping: {cur_url}")
                    try:
                        driver.back()
                        WebDriverWait(driver, SEARCH_WAIT).until(
                            EC.presence_of_element_located((By.CSS_SELECTOR, "a"))
                        )
                        randsleep(*DDG_SLEEP)
                    except Exception:
                        pass
                    continue

                if (
                    not abs_txt
                    and is_pdf_url(cur_url)
                    and not REJECT_PDFS
                    and USE_GROBID_FOR_PDFS
                ):
                    log.info(f"Detected PDF after click: {cur_url} — attempting GROBID")
                    abs_txt, note = process_pdf_with_grobid(cur_url, referer=url)

                if abs_txt:
                    abstract = abs_txt
                    url_used = cur_url
                    domain_last_hit[host] = time.time()
                    log.info(f"✅ ACCEPTED abstract via {note}, len={len(abstract)}")
                    driver.quit()
                    break
                else:
                    log.info(f"❌ No acceptable abstract on landing page ({note})")
                    if tried < len(ranked_seq):
                        try:
                            driver.back()
                            WebDriverWait(driver, SEARCH_WAIT).until(
                                EC.presence_of_element_located((By.CSS_SELECTOR, "a"))
                            )
                            randsleep(*DDG_SLEEP)
                        except Exception:
                            pass

            if abstract:
                break
            driver.quit()

        except WebDriverException as e:
            log.warning(f"Selenium error: {e}")
            try:
                driver.quit()
            except:
                pass
            break  # MAX_RETRIES=1 → abort immediately

    elapsed = round(time.time() - start, 2)
    return {
        "ddg_result_title": serp_title or pd.NA,
        "ddg_url": url_used or pd.NA,
        "abstract": abstract or pd.NA,
        "seconds": elapsed,
    }


# =========================
# DataFrame I/O (dtype-safe)
# =========================
def ensure_cols(df: pd.DataFrame) -> pd.DataFrame:
    if "abstract" not in df.columns:
        df["abstract"] = pd.Series(dtype="string")
    else:
        if not (
            pd.api.types.is_string_dtype(df["abstract"])
            or pd.api.types.is_object_dtype(df["abstract"])
        ):
            df["abstract"] = df["abstract"].astype("string")

    for col in ["ddg_result_title", "ddg_url"]:
        if col not in df.columns:
            df[col] = pd.Series(dtype="string")
        else:
            if not (
                pd.api.types.is_string_dtype(df[col])
                or pd.api.types.is_object_dtype(df[col])
            ):
                df[col] = df[col].astype("string")

    if "ddg_processed" not in df.columns:
        df["ddg_processed"] = pd.Series(dtype="boolean")
    else:
        if df["ddg_processed"].dtype != "boolean":
            df["ddg_processed"] = df["ddg_processed"].astype("boolean")

    return df


def save_df(df: pd.DataFrame, path: str):
    df.to_parquet(path, index=False)
    log.info(f"💾 Saved: {path}")


# =========================
# Main
# =========================
def main():
    log.info("=" * 80)
    log.info(
        "Start: Title → DDG → Click → Landing page → STRICT Abstract (heading-first + JSON-LD deep)"
    )
    log.info(
        f"PDF policy: REJECT_PDFS={REJECT_PDFS}, USE_GROBID_FOR_PDFS={USE_GROBID_FOR_PDFS}, GROBID_URL={GROBID_URL if USE_GROBID_FOR_PDFS else 'n/a'}"
    )
    log.info(f"RG policy: {RESEARCHGATE_POLICY}")
    log.info(f"Denylist: {sorted(DOMAIN_DENYLIST)}")
    log.info(f"STRICT_HOSTS: {sorted(STRICT_HOSTS)} (no meta desc, no generic CSS)")
    log.info(f"Input: {INPUT_PARQUET}")
    log.info(
        f"Save every: {SAVE_EVERY} rows | Delay: {DELAY_SEC}s | Retries: {MAX_RETRIES}"
    )
    log.info("=" * 80)

    if USE_GROBID_FOR_PDFS and REJECT_PDFS:
        log.info(
            "Note: USE_GROBID_FOR_PDFS=True but REJECT_PDFS=True → PDFs will be skipped. Set REJECT_PDFS=False to enable GROBID."
        )

    out_path = OUTPUT_PARQUET or INPUT_PARQUET.replace(
        ".parquet", "_with_ddg_abstracts.parquet"
    )

    df = pd.read_parquet(INPUT_PARQUET)

    # Force correct dtypes (prevents FutureWarning when assigning strings)
    df = ensure_cols(df)

    if TITLE_COL not in df.columns:
        log.error(f"Missing title column: {TITLE_COL}")
        log.info(f"Available columns: {list(df.columns)}")
        return

    mask_unprocessed = df["ddg_processed"].isna() | (df["ddg_processed"] == False)
    mask_need_title = df[TITLE_COL].notna() & (
        df[TITLE_COL].astype(str).str.strip() != ""
    )
    work_idx = df.index[mask_unprocessed & mask_need_title].tolist()

    # Backup after dtype fix to preserve the right schema
    backup = INPUT_PARQUET.replace(
        ".parquet", f"_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.parquet"
    )
    df.to_parquet(backup, index=False)
    log.info(f"Backup created: {backup}")

    log.info(f"Total rows: {len(df)} | To process now: {len(work_idx)}")

    processed_since_save = 0
    abstracts_found = int(df["abstract"].notna().sum())
    domain_last_hit: Dict[str, float] = {}

    with tqdm(
        total=len(work_idx), desc="🦆 DDG STRICT", unit="row", dynamic_ncols=True
    ) as pbar:
        for idx in work_idx:
            base_title = str(df.at[idx, TITLE_COL]).strip()
            fa = (
                first_author_family(df.at[idx, "author"])
                if "author" in df.columns
                else None
            )
            query = f"{base_title} {fa}" if fa else base_title

            res = process_title_with_retries(query, domain_last_hit)

            df.at[idx, "ddg_result_title"] = res["ddg_result_title"]
            df.at[idx, "ddg_url"] = res["ddg_url"]
            if pd.notna(res["abstract"]):
                df.at[idx, "abstract"] = res["abstract"]
                abstracts_found += 1
                log.info("✅ SUCCESS: abstract captured (strict)")
                log.info(
                    f"  Abstract Text: {res['abstract'][:200]}... ({len(res['abstract'])} chars)"
                )
            else:
                log.info("❌ No abstract captured (strict) [ok by policy]")

            df.at[idx, "ddg_processed"] = True
            processed_since_save += 1

            if processed_since_save >= SAVE_EVERY:
                save_df(df, out_path)
                processed_since_save = 0

            pbar.set_postfix(abs=abstracts_found)
            pbar.update(1)

            time.sleep(DELAY_SEC)

    save_df(df, out_path)

    log.info("=" * 80)
    log.info("DONE.")
    log.info(f"Abstracts total: {df['abstract'].notna().sum()}")
    log.info(f"Output: {out_path}")
    log.info(f"Log:    {LOG_FILE}")
    log.info("=" * 80)


if __name__ == "__main__":
    main()
