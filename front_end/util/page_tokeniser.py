from nltk.tokenize import RegexpTokenizer

# Words and fractions and percentages
tok = RegexpTokenizer(r'\d+[\.,]\d+%?|[\w%]+|=|≤|≥|<|>')


def tokenise_text(page: str) -> list:
    """
    Tokenise the content of a single page
    :param page:
    """
    for word in tok.tokenize(page):
        yield word

def tokenise_text_and_lowercase(page: str) -> list:
    """
    Tokenise the content of a single page and return the lowercase of each token
    :param page:
    """
    for word in tok.tokenize(page):
        yield word.lower()

def tokenise_pages(pages):
    """
    Tokenises the lists of pages and returns a list of lists.
    This is a custom tokeniser as nunbers and percentages should not be split (e.g. 5.5%).

    :param pages: List of strings which each correspond to the content of a single page.
    :return: List of lists of tokens.
    """
    tokenised_pages = []

    for page in pages:
        tokens = []
        for word in tokenise_text(page):
            tokens.append(word)
        tokenised_pages.append(tokens)
    return tokenised_pages



def iterate_tokens(tokenised_pages):
    for page_no, tokens in enumerate(tokenised_pages):
        for token_no, token in enumerate(tokens):
            yield page_no, token_no, token