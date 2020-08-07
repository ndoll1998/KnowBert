import unicodedata

def _is_punctuation(char):
    """Checks whether `char` is a punctuation character."""
    cp = ord(char)
    # We treat all non-letter/number ASCII as punctuation.
    # Characters such as "^", "$", and "`" are not in the Unicode
    # Punctuation class but we treat them as punctuation anyways, for
    # consistency.
    if (cp >= 33 and cp <= 47) or (cp >= 58 and cp <= 64) or (cp >= 91 and cp <= 96) or (cp >= 123 and cp <= 126):
        return True
    cat = unicodedata.category(char)
    if cat.startswith("P"):
        return True
    return False

def reconstruct_from_wordpieces(tokens):
    text, spans = "", []
    for t in tokens:
        begin = len(text)
        if '##' == t[:2]:
            # in word -> no whitespace
            text += t[2:]
        elif len(t) == 1 and _is_punctuation(t):
            # punctuation -> no whitespace
            text += t
        elif len(text) == 0:
            # beginning of text
            text += t
        else:
            # start new word -> add whitespace
            text += ' ' + t
            begin += 1
        # add token to spans
        end = len(text)
        spans.append((begin, end))
    # return text and spans
    return text, spans 