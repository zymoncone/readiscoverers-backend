import re
from bs4 import BeautifulSoup
from spellchecker import SpellChecker

from .constants import (
    CHAPTER_WITH_TITLE_PATTERN,
    TITLE_BEFORE_CHAPTER_PATTERN,
    PAGE_NUMBER_TAG_PATTERN,
    CHAPTER_ONLY_PATTERN,
)

spell = SpellChecker()


def find_book_meta_data(soup, meta_key, top_n=5):
    paragraphs = soup.find_all("p")[:top_n]

    book_title_pattern = re.compile(
        rf"^{meta_key}\s*[:：-]?\s*(?P<{meta_key}>.+)$", re.IGNORECASE
    )

    for p in paragraphs:
        text = p.get_text(strip=True)
        match = book_title_pattern.match(text)
        if match:
            return match.group(meta_key)
    return None


def title_case(func):
    def wrapper(*args, **kwargs):
        chapter_info = func(*args, **kwargs)
        if chapter_info:
            text = chapter_info["title"]
            words = text.split()
            titled_text = []

            for word in words:
                hyphenated_words = word.split("-")
                if len(hyphenated_words) > 1:
                    titled_hyphenated = [
                        (
                            h_word[0].upper() + h_word[1:].lower()
                            if len(h_word) > 1
                            else h_word.upper()
                        )
                        for h_word in hyphenated_words
                    ]
                    titled_text.append("-".join(titled_hyphenated))
                else:
                    titled_text.append(
                        word[0].upper() + word[1:].lower()
                        if len(word) > 1
                        else word.upper()
                    )
            return {
                "title": " ".join(titled_text),
                "sourceline": chapter_info["sourceline"],
            }
        return chapter_info

    return wrapper


def remove_page_number_hyperlinks(element: BeautifulSoup):
    element_copy = element.__copy__()

    # Remove standalone <a> tags with page numbers
    for a in element_copy.find_all("a"):
        if a.get_text(strip=True).isdigit() or PAGE_NUMBER_TAG_PATTERN.match(
            a.get_text(strip=True)
        ):
            a.decompose()

    # Remove <span> tags containing page number links
    for span in element_copy.find_all("span"):
        if a := span.find("a"):
            if a.get_text(strip=True).isdigit() or PAGE_NUMBER_TAG_PATTERN.match(
                a.get_text(strip=True)
            ):
                span.decompose()

    return element_copy


def match_chapter_title(text, header) -> dict:

    if match := CHAPTER_WITH_TITLE_PATTERN.match(text):
        return {"title": match.group("title").strip(), "sourceline": header.sourceline}

    if match := TITLE_BEFORE_CHAPTER_PATTERN.match(text):
        return {"title": match.group("title").strip(), "sourceline": header.sourceline}

    # If h2 only contains chapter number, look for subtitle in next sibling
    if CHAPTER_ONLY_PATTERN.match(text):
        if chapter_info := extract_subtitle(header, only_chapter_number_found=True):
            return chapter_info

    return None


def get_img_text(element: BeautifulSoup, title_only: bool = False) -> str | None:
    """Extract alt or title text from an img tag within an element."""
    if img := element.find("img"):
        if img.get("alt") and not title_only:
            return img.get("alt").strip()
        elif img.get("title"):
            return img.get("title").strip()
    return None


@title_case
def extract_chapter_title_from_paragraph(
    p_element: BeautifulSoup, chapter_id_keyword: str = "chap"
) -> dict | None:
    if (
        p_element.find("a")
        and chapter_id_keyword in p_element.find("a").get("id", "").lower()
    ):
        if chapter_info := extract_subtitle(
            p_element, element_class=p_element.get("class", [])
        ):
            return chapter_info
    return None


@title_case
def extract_chapter_title_from_div(
    div_element: BeautifulSoup, chapter_class_keyword: str = "chap"
) -> dict | None:
    if chapter_class_keyword in div_element.get("class", []):
        if chapter_title := div_element.find("span").get_text().strip():
            return {"title": chapter_title, "sourceline": div_element.sourceline}
    return None


@title_case
def extract_chapter_title_from_header(
    h2, book_title: str = None, book_author: str = None
) -> dict | None:
    # 1. Check for chapter wording in header text
    text = remove_page_number_hyperlinks(h2).get_text(" ", strip=True)

    if any(
        keyword in text.lower()
        for keyword in [
            "project gutenberg",
            "list of chapters",
        ]
    ):
        return None

    if (
        text.lower() == "contents"
        or text.lower() == "the end"
        or (book_title and text.lower() == book_title.lower())
        or (book_author and book_author.lower() in text.lower())
    ):
        return None

    # Check if there's a paragraph before the next h1/h2
    next_p = h2.find_next_sibling("p")
    next_header = h2.find_next_sibling(["h1", "h2"])

    # If there's no paragraph, or the next header comes before the paragraph, skip
    if not next_p or (next_header and next_header.sourceline < next_p.sourceline):
        # print("skipping h2 with no paragraph sibling")
        return None

    if chapter_info := match_chapter_title(text, h2):
        # print("matched pattern 1 from h2")
        return chapter_info

    # 2. Check for img tag in h2
    if img_text := get_img_text(h2):
        if chapter_info := match_chapter_title(img_text, h2):
            # print("matched pattern 2 from h2")
            return chapter_info

    return {"title": text, "sourceline": h2.sourceline}


def extract_subtitle(
    element: BeautifulSoup,
    only_chapter_number_found: bool = False,
    element_class: list = None,
) -> dict | None:
    # iterate through the next siblings
    for sibling in element.find_next_siblings():

        # stop if we hit another header, aka next chapter
        if sibling.name in ("h1", "h2"):
            return None

        # Check for p with element_class match first (for paragraph-based chapters)
        if (
            sibling.name == "p"
            and element_class
            and (set(element_class) & set(sibling.get("class", [])))
        ):
            if text := remove_page_number_hyperlinks(sibling).get_text(strip=True):
                return {"title": text, "sourceline": sibling.sourceline}

        # Check for p with h3 class (this should come BEFORE the h3 tag check)
        if sibling.name == "p" and "h3" in sibling.get("class", []):
            if text := remove_page_number_hyperlinks(sibling).get_text(strip=True):
                return {"title": text, "sourceline": sibling.sourceline}

        # check for img tag in sibling
        if img_text := get_img_text(sibling):
            return {"title": img_text, "sourceline": sibling.sourceline}

        # if only chapter number found, look for h3 as subtitle
        if only_chapter_number_found and sibling.name == "h3":
            if text := "".join(sibling.find_all(string=True, recursive=False)).strip():
                return {"title": text, "sourceline": sibling.sourceline}
    return None


# paragraph extraction functions


def fix_split_words(text: str) -> str:
    """Fix words that were split by removed HTML tags.
    Combines words when at least one part is invalid individually
    but they form a valid word when combined."""
    words = text.split()
    fixed_words = []
    i = 0

    while i < len(words):
        current_word = words[i]

        # Check if there's a next word to potentially combine with
        if i + 1 < len(words):
            next_word = words[i + 1]
            combined = current_word + next_word

            # Check validity
            current_valid = current_word.lower() in spell
            next_valid = next_word.lower() in spell
            combined_valid = combined.lower() in spell

            # Combine if:
            # 1. At least one part is invalid on its own
            # 2. Combined word IS valid
            if (not current_valid or not next_valid) and combined_valid:
                fixed_words.append(combined)
                i += 2
                continue

        # Don't combine - keep the word as is
        fixed_words.append(current_word)
        i += 1

    return " ".join(fixed_words)


def get_paragraph_with_dropcap(p_element):
    """Extract paragraph text, including drop cap from preceding img if present."""

    prev_sibling = p_element.find_previous_sibling()

    if prev_sibling and prev_sibling.name == "div":
        if drop_cap := get_img_text(prev_sibling, title_only=True):
            text = remove_page_number_hyperlinks(p_element).get_text(
                " ", strip=True
            )  # Use space separator
            return fix_split_words(drop_cap.strip() + text)
        if drop_cap := prev_sibling.get("title"):
            text = remove_page_number_hyperlinks(p_element).get_text(
                " ", strip=True
            )  # Use space separator
            return fix_split_words(drop_cap.strip() + text)

    if img := p_element.find("img"):
        if drop_cap := img.get("alt"):
            p_copy = p_element.__copy__()
            p_copy.find("img").decompose()
            text = remove_page_number_hyperlinks(p_copy).get_text(
                " ", strip=True
            )  # Use space separator
            return fix_split_words(drop_cap.strip() + text)

    p_classes = p_element.get("class", [])
    if any("drop" in cls.lower() for cls in p_classes):
        text = remove_page_number_hyperlinks(p_element).get_text(
            " ", strip=True
        )  # Use space separator
        return fix_split_words(text)

    text = remove_page_number_hyperlinks(p_element).get_text(
        " ", strip=True
    )  # Use space separator
    return fix_split_words(text)


def extract_chapter_content(
    p_elements: list[BeautifulSoup], chapters: list[dict]
) -> list[dict]:
    """Extract paragraph text for each chapter based on sourceline positions."""

    for i, chapter in enumerate(chapters):
        # Skip extraction if content already exists
        if chapter.get("content"):
            chapter["content"] = chapter["content"]
            chapter["paragraph_count"] = chapter["paragraph_count"]
            continue

        current_chapter_sourceline = chapter["sourceline"]
        next_sourceline = (
            chapters[i + 1]["sourceline"] if i + 1 < len(chapters) else None
        )

        # Filter paragraphs between this chapter and the next
        chapter_paragraphs = []
        for p in p_elements:
            # Skip paragraphs before the current chapter
            if p.sourceline <= current_chapter_sourceline:
                continue

            # Stop if we've reached the next chapter
            if next_sourceline and p.sourceline >= next_sourceline:
                break

            # Extract and clean the paragraph text
            if text := get_paragraph_with_dropcap(p):
                if (
                    CHAPTER_ONLY_PATTERN.match(text)
                    or text.lower() == "view image"
                    or ("chap" in text.lower() and len("chapter 00") >= len(text))
                ):
                    continue
                chapter_paragraphs.append(text.replace("\n", " "))

        # Add content to chapter
        chapter["content"] = "\n\n".join(chapter_paragraphs)
        chapter["paragraph_count"] = len(chapter_paragraphs)

    return chapters


def extract_preface(soup, start_sourceline, first_chapter_sourceline):
    """Extract preface if there's a drop cap paragraph or 'To my readers' before the first chapter."""
    if not first_chapter_sourceline:
        return None

    # Look for paragraphs with drop cap class or "To my readers" before first chapter
    for p in soup.find_all("p"):
        # Skip paragraphs before the book starts
        if start_sourceline and p.sourceline < start_sourceline:
            continue

        # Stop if we've reached the first chapter
        if p.sourceline >= first_chapter_sourceline:
            break

        # Check if paragraph has drop cap class or starts with "To my readers"
        p_classes = p.get("class", [])
        text = get_paragraph_with_dropcap(p)

        if any("drop" in cls.lower() for cls in p_classes) or (
            text
            and text.lower().startswith("to my readers")
            or (text and text.lower().startswith("dear boys and girls"))
        ):
            # Found a valid paragraph - extract content until next non-p element
            content_paragraphs = []
            current = p

            while current:
                if current.name == "p":
                    if para_text := get_paragraph_with_dropcap(current):
                        content_paragraphs.append(para_text.replace("\n", " "))
                    current = current.find_next_sibling()
                else:
                    # Hit a non-p element, stop
                    break

            if content_paragraphs:
                return {
                    "title": "Preface",
                    "content": "\n\n".join(content_paragraphs),
                    "paragraph_count": len(content_paragraphs),
                }

    return None


def check_sourceline_bounds(elements, start_sourceline, end_sourceline):
    for element in elements:
        if start_sourceline and element.sourceline < start_sourceline:
            continue
        if end_sourceline and element.sourceline > end_sourceline:
            break
        yield element


def parse_html_book(html_file: str) -> dict:
    soup = BeautifulSoup(html_file, "html.parser")

    book_title = find_book_meta_data(soup, "title")
    book_author = find_book_meta_data(soup, "author")

    start_sourceline = start_tag.sourceline if (start_tag := soup.find("h1")) else None
    end_sourceline = (
        end_tag.sourceline
        if (end_tag := soup.find("img", alt=re.compile(r"the end", re.IGNORECASE)))
        else None
    )

    chapters = []
    chapter_index = 1

    # Find first chapter's sourceline to use as boundary
    first_chapter_sourceline = None
    for header in soup.find_all("h2"):
        if chapter_info := extract_chapter_title_from_header(
            header, book_title, book_author
        ):
            first_chapter_sourceline = header.sourceline
            break

    for paragraph in soup.find_all("p"):
        if chapter_info := extract_chapter_title_from_paragraph(paragraph):
            first_chapter_sourceline = paragraph.sourceline
            break

    for division in soup.find_all("div"):
        if chapter_info := extract_chapter_title_from_div(division):
            first_chapter_sourceline = division.sourceline
            break

    # Extract any preface text before chapters that won't be captured by title headers
    if preface_section := extract_preface(
        soup, start_sourceline, first_chapter_sourceline
    ):
        chapters.append(preface_section)

    for header in check_sourceline_bounds(
        soup.find_all("h2"), start_sourceline, end_sourceline
    ):
        if chapter_info := extract_chapter_title_from_header(
            header, book_title, book_author
        ):
            chapters.append(chapter_info)

    for paragraph in check_sourceline_bounds(
        soup.find_all("p"), start_sourceline, end_sourceline
    ):
        if chapter_info := extract_chapter_title_from_paragraph(paragraph):
            chapters.append(chapter_info)

    for division in check_sourceline_bounds(
        soup.find_all("div"), start_sourceline, end_sourceline
    ):
        if chapter_info := extract_chapter_title_from_div(division):
            chapters.append(chapter_info)

    if not chapters:
        # no chapters found
        return None

    for ch in chapters:
        starts_with_keyword = (
            ch["title"].lower().startswith("by")
            or ch["title"].lower().startswith("author")
            or ch["title"].lower().startswith("dedicated to")
        )
        if (
            ch["title"].lower()
            in [
                "to our readers",
                "to my readers",
                "introduction",
                "prologue",
                "preface",
            ]
            or starts_with_keyword
        ):
            chapter_index = 0
        ch.update({"index": chapter_index})
        chapter_index += 1

    chapters = extract_chapter_content(soup.find_all("p"), chapters)

    return {"title": book_title, "author": book_author, "chapters": chapters}
