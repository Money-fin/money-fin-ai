import re
import unicodedata


def preprocess(sentence):
    sentence = _remove_bracket(sentence)
    sentence = _remove_punctuations(sentence)
    sentence = _replace_whitespace(sentence)

def _remove_bracket(sentence):
    sentence = re.sub(r"\[[a-zA-Z0-9가-힣\s]*\]\s*", "", sentence)
    sentence = re.sub(r"\([a-zA-Z0-9가-힣\s]*\)\s*", "", sentence)
    return sentence

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
    x = normalize(x, "NFD")
    x = list(x)

    x = []
    for each in x:
        if _ispunctuation(each) is True:
            continue
        x.append(each)
    x = normalize(''.join(x), "NFC")
    return x

def _replace_whitespace(sentence):
    sentence = re.sub(r"\s+", " ", sentence)
    return sentence

# print(remove_bracket("[] 아아"))
# print(remove_bracket("[아아아 1아] 아아"))
# print(_remove_bracket("(아아아 1아) 아아"))
# print(remove_bracket("[아아아 1아]아아"))
# print(remove_bracket("[아아아1아] 아아"))
