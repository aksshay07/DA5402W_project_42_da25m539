def clean_text(text: str) -> str:
    import re
    text = text.lower()
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"[^a-z0-9\s]", "", text)
    return text.strip()


def test_lowercasing():
    assert clean_text("Hello World") == "hello world"


def test_url_removal():
    assert "http" not in clean_text("check this https://example.com out")


def test_punctuation_removal():
    result = clean_text("wow!!! so great :)")
    assert "!" not in result and ":" not in result and ")" not in result


def test_empty_string_passthrough():
    assert clean_text("") == ""


def test_numbers_preserved():
    assert "42" in clean_text("I have 42 apples")