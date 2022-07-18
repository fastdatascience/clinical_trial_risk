import base64
import bz2
import operator
import pickle as pkl
import re
import time
from collections import Counter
from io import BytesIO
from os.path import exists

import numpy as np
from PIL import Image
from nltk.corpus import stopwords
from wordcloud import WordCloud

stops = set(stopwords.words('english')).union(set(stopwords.words('french')))
stops.add("protocol")
stops.add("protocols")
stops.add("subject")
stops.add("subjects")
stops.add("trial")
stops.add("trials")
stops.add("doctor")
stops.add("doctors")
stops.add("eg")
stops.add("get")
stops.add("getting")
stops.add("got")
stops.add("gotten")
stops.add("rx")

MAX_NUM_WORDS = 100

word_cloud_token_regex = re.compile(r'(?i)^([a-z][a-z]+)$')


class WordCloudGenerator:

    def __init__(self, path_to_classifier):
        """
        Load the default values of IDF for the words in a training corpus from a Pickle file. This is needed to determine the size of the words in the word cloud as words which occur in many protocols are not informative and should appear small so the word cloud algorithm is a little different from the standard word cloud algorithm which is based on frequency alone

        :param path_to_classifier: The path to the Pickle file containing IDFs
        """
        if not exists(path_to_classifier):
            print(
                f"WARNING! UNABLE TO LOAD WORD CLOUD IDFS FILE {path_to_classifier}. You need to run the training script.")
            self.idf = dict()
            self.idf[""] = np.log(20 / 1)
            return
        with bz2.open(path_to_classifier, "rb") as f:
            self.idf = pkl.load(f)

    def generate_word_cloud(self, tokenised_pages: list, condition_to_pages: dict):
        """
        Giving a list of tokenized pages this function generates a word cloud using Matplotlib and outputs in as base64 encoded image
        Terms which were decisive in the decision of the pathology classifier are displayed in a different colour, hence why the condition dictionary is needed

        :param tokenised_pages: Tokens occurring in the document by page
        :param condition_to_pages: Needed to identify which terms contributed to the decision of the pathology
        :return:
        """
        start_time = time.time()

        def condition_colour_func(word, font_size, position, orientation, random_state=None, **kwargs):
            if word.lower() in condition_to_pages.get("terms", set()):
                # return 'rgb(56,163,165)' # FDS
                return 'rgb(246, 78, 139)'
            return "rgb(50,50,50)"

        tfs = Counter()
        normalised_to_all_surface_forms = {}
        token_page_occurrences = Counter()
        for page_tokens in tokenised_pages:
            unique_tokens_on_page = set()
            for token_idx, token in enumerate(page_tokens):
                if word_cloud_token_regex.match(token):
                    tl = token.lower()
                    if tl in stops:
                        continue
                    unique_tokens_on_page.add(tl)
                    tfs[tl] += 1
                    if tl not in normalised_to_all_surface_forms:
                        normalised_to_all_surface_forms[tl] = Counter()
                    normalised_to_all_surface_forms[tl][token] += 1
            for tl in unique_tokens_on_page:
                token_page_occurrences[tl] += 1

        tf_idf = {}
        for term, tf in tfs.items():
            # Ignore any term that occurs on more than half the pages
            # These are often rubbish
            # Unless they are informative terms such as HIV.
            if token_page_occurrences[term] > len(tokenised_pages) / 4 and term not in condition_to_pages.get("terms",
                                                                                                              set()):
                continue
            canonical = \
                sorted(normalised_to_all_surface_forms[term].items(), key=operator.itemgetter(1), reverse=True)[0][0]
            tf_idf[canonical] = tf * self.idf.get(term, self.idf.get(""))

        tf_idfs_to_include = sorted(tf_idf.items(), key=operator.itemgetter(1), reverse=True)
        if len(tf_idfs_to_include) > MAX_NUM_WORDS:
            tf_idfs_to_include = tf_idfs_to_include[:MAX_NUM_WORDS]

        word_cloud_sizes = {}
        tf_idfs_to_include.reverse()
        for counter, (term, tf_idf) in enumerate(tf_idfs_to_include):
            word_cloud_sizes[term] = counter + 1

        wordcloud = WordCloud(width=900, height=400,
                              background_color='white', color_func=condition_colour_func).generate_from_frequencies(
            word_cloud_sizes)

        img = wordcloud.to_array()

        pil_img = Image.fromarray(img)
        buff = BytesIO()
        pil_img.save(buff, format="PNG")

        encoded = base64.b64encode(buff.getvalue()).decode("ascii")

        end_time = time.time()

        log_message = f"Word cloud generated in {end_time - start_time:.2f} seconds."
        print(log_message)

        return 'data:image/png;base64,{}'.format(encoded), log_message
