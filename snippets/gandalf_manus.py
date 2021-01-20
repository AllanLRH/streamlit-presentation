# %% setup
import requests
import re
from typing import List, Generator, Optional
import time
from loguru import logger


logger.remove()
# logger.add("gandalf.log", level="TRACE", format="{level}: {message}")
logger.add("gandalf.log", level="DEBUG", format="{level}: {message}")
logger.add("gandalf.log", level="INFO", format="{level}: {message}")

# %%


def yield_gandalf(corpus: List[str]) -> Generator[str, None, None]:
    rx_character = re.compile(r"^ {20,21}[A-Z]+$")
    rx_speech = re.compile(r"^ {10}\S.+")

    is_speech = False
    is_gandalf = False

    to_yield = list()
    for i, line in enumerate(corpus):
        if not line.strip():
            logger.trace(f"Skipping empty line {i}")
            continue
        line = line.rstrip()

        logger.trace(f"Processing line {i}: {line!r}")
        is_character = bool(rx_character.match(line))
        if is_character:
            is_gandalf = line.endswith("GANDALF")
        is_speech = bool(rx_speech.match(line))
        logger.trace(f"{is_character=} {is_gandalf=} {is_speech=}")
        if is_gandalf and is_character:
            logger.info(f"Entered a gandalf block on line {i}")
        if is_speech and is_gandalf:
            # to_yield.append(f"({i}): {line.strip()}")
            to_yield.append(line.strip())
        elif is_gandalf and not is_speech and to_yield:
            concatenated = " ".join(to_yield)
            to_yield = list()
            yield concatenated


def get_gandalf_lines(book: str, delay: float = 0) -> List[str]:
    """
    Book must be one of
      * The Fellowship of the Ring
      * The Two Towers
      * The Return of the King

    """
    book_options = {
        "The Fellowship of the Ring": (
            "assets/lotr-01.txt",
            "https://raw.githubusercontent.com/eDubrovsky/movie_scripts/master/Lord-of-the-Rings-Fellowship-of-the-Ring%2C-The.txt",
        ),
        "The Two Towers": (
            "assets/lotr-02.txt",
            "https://raw.githubusercontent.com/eDubrovsky/movie_scripts/master/Lord-of-the-Rings-The-Two-Towers.txt",
        ),
        "The Return of the King": (
            "assets/lotr-03.txt",
            "https://raw.githubusercontent.com/eDubrovsky/movie_scripts/master/Lord-of-the-Rings-Return-of-the-King.txt",
        ),
    }

    if book not in book_options:
        raise KeyError(f"`book` must be one of {book_options.keys()}, but was {book!r}")

    filepath, url = book_options[book]
    try:
        1 / 0  # TODO: Remove this
        r = requests.get(url)
        raw = r.text.splitlines()
    except Exception:
        with open(filepath) as fid:
            raw = fid.read().splitlines()

    time.sleep(delay)

    return list(yield_gandalf(raw))


if __name__ == "__main__":
    book_title = "The Fellowship of the Ring"
    corpus = get_gandalf_lines(book_title, delay=3)
    print(f"{len(corpus)} lines in corpus for the book {book_title}", end="\n\n")
    print(*corpus[:10], sep="\n")
