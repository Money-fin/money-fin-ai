import re
import unicodedata

def _normalize(s, form="NFD"):
    return unicodedata.normalize(form, s)


def remove_controls(x):
    x = re.sub('\s+', " ", x)
    return x


def remove_bracket(x):
    def _remove_punctuations(x):
        def _is_punctuation(char):
            cp = ord(char)
            if (
                    (33 <= cp <= 47)
                    or (58 <= cp <= 64)
                    or (91 <= cp <= 96)
                    or (123 <= cp <= 126)
            ):
                return True
            cat = unicodedata.category(char)
            if cat.startswith("P"):
                return True
            return False

        x = _normalize(x, "NFD")
        x = list(x)

        x_ = []
        for each in x:
            if _is_punctuation(each) is True:
                continue
            x_.append(each)
        x = _normalize(''.join(x_), "NFC")
        return x

    x = re.sub(r"\[[a-zA-Z0-9가-힣\s]*\]\s*", "", x)
    x = re.sub(r"\([a-zA-Z0-9가-힣\s]*\)\s*", "", x)
    x = _remove_punctuations(x)
    return x


def replace_whitespace(x):
    x = re.sub(r"\s+", " ", x)
    return x


f_list = [
    remove_bracket,
    remove_controls,
    replace_whitespace,
    lambda x: x.lower(),
    lambda x: re.sub(r"[^a-zA-Z가-힣ㄱ-ㅎ?.!,0-9]", " ", x),
]
