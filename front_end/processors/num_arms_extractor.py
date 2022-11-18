import spacy
from spacy.matcher import Matcher

from util.spacy_wrapper import nlp

word2num = {'one': 1,
            'two': 2,
            'three': 3,
            'four': 4,
            'five': 5,
            'six': 6,
            'seven': 7,
            'eight': 8,
            'nine': 9,
            'ten': 10,
            'eleven': 11,
            'twelve': 12,
            'thirteen': 13,
            'fourteen': 14,
            'fifteen': 15,
            'sixteen': 16,
            'seventeen': 17,
            'eighteen': 18,
            'nineteen': 19,
            'both': 2,
            'single': 2}
number_words = list(word2num)
for n in range(1, 20):
    word2num[str(n)] = n
matcher = Matcher(nlp.vocab)
patterns = [
    # [{"LIKE_NUM":True},  {"LOWER": {"IN": ["treatment", "study", "dose"]}, "OP":"?"}, {"LOWER": {"IN": ["arm", "arms", "group", "groups", "subgroup", "subgroups", "cohort", "cohorts"]}}],
    [{"LOWER": {"IN": number_words}}, {"LOWER": {"IN": ["treatment", "study", "dose", "experimental", "dosage"]}},
     {"LOWER": {"IN": ["arm", "arms", "group", "groups", "subgroup", "subgroups", "cohort", "cohorts"]}}],
    #            [{"LOWER":{"IN":number_words}},  {"LOWER": {"IN": ["group", "groups", "subgroup", "subgroups", "cohort", "cohorts"]}}],
    [{"LOWER": {"IN": list(word2num)}}, {"LOWER": "-", "OP": "?"}, {"LOWER": {"IN": ["armed"]}}]
]
matcher.add("arms", patterns)


class NumArmsExtractor:

    def process(self, tokenised_pages: list) -> tuple:
        """
        Identify the number of arms.
        :param tokenised_pages: List of lists of tokens of each page.
        :return: The prediction (str) and a map from number of arms to the pages it's mentioned in.
        """

        tokenised_pages = [[string.lower() for string in sublist] for sublist in tokenised_pages]

        num_arms_to_pages = {}
        context = {}

        for page_number, page_tokens in enumerate(tokenised_pages):
            doc = spacy.tokens.doc.Doc(
                nlp.vocab, words=page_tokens)
            matches = matcher(doc)
            for word, start, end in matches:
                matching_span = doc[start:end]
                if matching_span not in num_arms_to_pages:
                    num_arms_to_pages[matching_span] = []
                num_arms_to_pages[matching_span].append(page_number)

                text = matching_span.text.lower()
                sentence = doc[max(0, start - 5):min(len(doc), end + 5)].text
                if text not in context:
                    context[text] = ""
                context[text] = (context[text] + " " + f"Page {page_number + 1}: " + sentence).strip()

        num_arms_to_pages = sorted(num_arms_to_pages.items(), key=lambda v: len(v[1]), reverse=True)

        prediction = None
        # if len(num_arms_to_pages) == 1:
        if len(num_arms_to_pages) == 1 or \
                (len(num_arms_to_pages) > 1 and len(num_arms_to_pages[0][1]) > len(num_arms_to_pages[1][1])):
            for matching_span in num_arms_to_pages[0][0]:
                if matching_span.text in word2num:
                    prediction = word2num[matching_span.text]
                    break
        if prediction is not None and prediction > 5:
            prediction = 5

        num_arms_text_to_pages = {}
        for phrase, value in num_arms_to_pages:
            if phrase.text.lower() not in num_arms_to_pages:
                num_arms_text_to_pages[phrase.text.lower()] = []
            num_arms_text_to_pages[phrase.text.lower()].extend(value)

        return {"prediction": prediction, "pages": num_arms_text_to_pages, "context": context}
