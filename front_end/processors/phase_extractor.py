import spacy
from spacy.matcher import PhraseMatcher

nlp = spacy.load('en_core_web_sm', disable=['ner', 'tagger', 'parser', 'lemmatizer'])

phase_synonyms = {1: ['phase i', 'phase 1 b', 'phase 1', 'phase 1.0'],
                  1.5: ['phase i ii', 'phase 1 2', 'phase 1 2 a'],
                  2: ['phase ii', 'phase 2 b', 'phase 2', 'phase 2.0'],
                  2.5: ['phase 2.5', 'phase ii iii'],
                  3: ['phase iii', 'phase 3', 'phase 3.0'],
                  }

phrase_matcher = PhraseMatcher(nlp.vocab)

for phase_number, synonyms in phase_synonyms.items():
    phases = [nlp.make_doc(text) for text in synonyms]

    phrase_matcher.add(f"Phase {phase_number}", None, *phases)


class PhaseExtractor:

    def process(self, tokenised_pages: list) -> tuple:
        """
        Identify the trial phase.
        :param tokenised_pages: List of lists of tokens of each page.
        :return: The prediction (str) and a map from phase to the pages it's mentioned in.
        """

        tokenised_pages = [[string.lower() for string in sublist] for sublist in tokenised_pages]

        phase_to_pages = {}

        for page_number, page_tokens in enumerate(tokenised_pages):
            doc = spacy.tokens.doc.Doc(
                nlp.vocab, words=page_tokens)
            phrase_matches = phrase_matcher(doc)
            for word, start, end in phrase_matches:
                phase_number = nlp.vocab.strings[word]
                if phase_number not in phase_to_pages:
                    phase_to_pages[phase_number] = []
                phase_to_pages[phase_number].append(page_number)

        phase_to_pages = sorted(phase_to_pages.items(), key=lambda v: len(v[1]), reverse=True)

        prediction = 0
        if len(phase_to_pages) > 0:
            prediction = float(phase_to_pages[0][0].split(" ")[1])

        return {"prediction": prediction, "pages": dict(phase_to_pages)}

        # return {"prediction": prediction, "pages": []}
