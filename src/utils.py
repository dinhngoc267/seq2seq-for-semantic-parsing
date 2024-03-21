import re


# Load tokenize function
def normalize_cased_token(token):
    flag = False
    for c in token:
        if c.isupper():
            flag = True
            break
    if flag:
        token = token.lower()
        token = token[0].upper() + token[1:]

    return token


def tokenize_sequence(text):
    tokens = re.findall(r"\w+|[^\w\s]", text.lower(), re.UNICODE)
    # normalized_tokens = [normalize_cased_token(token) for token in tokens]

    return tokens