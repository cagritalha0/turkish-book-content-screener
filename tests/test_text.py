from bookscreener.text import (
    atesman_band,
    atesman_score,
    clean_social_text,
    count_syllables,
    split_sentences,
    tokenize_words,
    turkish_lower,
)


def test_turkish_lower_handles_dotted_and_dotless_i():
    assert turkish_lower("IRMAK") == "ırmak"
    assert turkish_lower("İSTANBUL") == "istanbul"
    assert "̇" not in turkish_lower("İ")  # no combining dot left behind


def test_clean_social_text_masks_users_and_urls():
    cleaned = clean_social_text("@USER Bak şuna https://t.co/xyz #Harika")
    assert cleaned == "<user> bak şuna <url> harika"


def test_tokenize_words_drops_punctuation():
    assert tokenize_words("Ama, ADAM geldi!") == ["ama", "adam", "geldi"]


def test_count_syllables():
    assert count_syllables("kitap") == 2
    assert count_syllables("KELEBEK") == 3  # upper-case vowels count too


def test_atesman_ignores_case_and_empty_sentences():
    text = "Ali topu at. Ali koş."
    assert atesman_score(text) == atesman_score(text.upper())
    assert atesman_score("") == 0.0


def test_atesman_simple_text_is_easy():
    assert atesman_band(atesman_score("Ali top at. Ayşe ip atla. Kedi süt iç.")) in {"easy", "very easy"}


def test_split_sentences():
    assert split_sentences("Merhaba çocuklar. Bugün masal var! Hazır mısınız?") == [
        "Merhaba çocuklar.",
        "Bugün masal var!",
        "Hazır mısınız?",
    ]
    assert split_sentences("   ") == []
