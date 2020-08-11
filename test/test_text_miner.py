import pandas as pd
import unittest

from easyexplore.data_explorer import DataExplorer
from easyexplore.text_miner import TextMiner

DATA_SET: pd.DataFrame = pd.read_csv(filepath_or_buffer='amazon_musical_instruments_reviews.csv')
ID_TEXT: dict = DataExplorer(df=DATA_SET).get_feature_types()
TEXT_MINER: TextMiner = TextMiner(df=DATA_SET,
                                  features=ID_TEXT.get('id_text'),
                                  lang='en',
                                  auto_interpret_natural_language=True
                                  )


class TextMinerTest(unittest.TestCase):
    """
    Unit test for class TextMiner
    """
    def test_clustering(self):
        pass

    def test_detect_lang(self):
        _lang_feature: int = len(TEXT_MINER.get_str_match(cases=list(TEXT_MINER.df.keys()), substring='_lang'))
        TEXT_MINER.detect_lang(sampling=True)
        self.assertTrue(expr=_lang_feature == 0 and len(TEXT_MINER.get_str_match(cases=list(TEXT_MINER.df.keys()), substring='_lang')) > 0)

    def test_emoji_handler(self):
        _emoji_feature: int = len(TEXT_MINER.get_str_match(cases=list(TEXT_MINER.df.keys()), substring='_emojis'))
        TEXT_MINER.emoji_handler()
        self.assertTrue(expr=_emoji_feature == 0 and len(TEXT_MINER.get_str_match(cases=list(TEXT_MINER.df.keys()), substring='_emojis')) > 0)

    def test_find_occurances(self):
        _occurance_feature: int = len(TEXT_MINER.get_str_match(cases=list(TEXT_MINER.df.keys()), substring='_MONEY'))
        TEXT_MINER.count_occurances(features=['reviewText'], search_text='_MONEY')
        self.assertTrue(expr=_occurance_feature == 0 and len(TEXT_MINER.get_str_match(cases=list(TEXT_MINER.df.keys()), substring='_MONEY')) > 0)

    def test_generate_linguistic_features(self):
        _ner_feature: int = len(TEXT_MINER.get_str_match(cases=list(TEXT_MINER.df.keys()), substring='_ner_'))
        _pos_feature: int = len(TEXT_MINER.get_str_match(cases=list(TEXT_MINER.df.keys()), substring='_pos_'))
        TEXT_MINER.generate_linguistic_features()
        self.assertTrue(expr=_ner_feature == 0 and _pos_feature == 0 and (len(TEXT_MINER.get_str_match(cases=list(TEXT_MINER.df.keys()), substring='_ner')) > 0) and (len(TEXT_MINER.get_str_match(cases=list(TEXT_MINER.df.keys()), substring='_pos')) > 0))

    def test_get_generated_features(self):
        self.assertTrue(expr=TEXT_MINER.get_generated_features().shape[1] > 0)

    def test_get_str_match(self):
        self.assertTrue(expr=len(TEXT_MINER.get_str_match(cases=['abc', 'def'], substring='b')) > 0)

    def test_merge(self):
        _merge_feature: int = len(TEXT_MINER.get_str_match(cases=list(TEXT_MINER.df.keys()), substring='_merge_'))
        TEXT_MINER.merge(features=['reviewText', 'summary'])
        self.assertTrue(expr=_merge_feature == 0 and len(TEXT_MINER.get_str_match(cases=list(TEXT_MINER.df.keys()), substring='_merge_')) > 0)

    def test_replace(self):
        TEXT_MINER.replace(features=['summary'], find_values=['MONEY'], replace_value='Gianni')
        TEXT_MINER.count_occurances(features=['summary'], search_text='Gianni')
        self.assertTrue(expr=len(TEXT_MINER.get_str_match(cases=list(TEXT_MINER.df.keys()), substring='summary_Gianni')) > 0)

    def test_splitter(self):
        _split_feature: int = len(TEXT_MINER.get_str_match(cases=list(TEXT_MINER.df.keys()), substring='_split_'))
        TEXT_MINER.splitter(features=['reviewText'], sep=' ')
        self.assertTrue(expr=_split_feature == 0 and len(TEXT_MINER.get_str_match(cases=list(TEXT_MINER.df.keys()), substring='_split_')) > 0)

    def test_segmentation(self):
        self.assertDictEqual(d1={'enumeration': [],
                                 'phrases': ['reviewerName', 'reviewText', 'summary'],
                                 'id': [],
                                 'email': [],
                                 'rating': [],
                                 'url': [],
                                 'unknown': ['reviewerID', 'asin']
                                 },
                             d2=TEXT_MINER.segments
                             )

    def test_similarity(self):
        _similarity_feature: int = len(TEXT_MINER.get_str_match(cases=list(TEXT_MINER.df.keys()), substring='_similarity'))
        TEXT_MINER.similarity()
        self.assertTrue(expr=_similarity_feature == 0 and len(TEXT_MINER.get_str_match(cases=list(TEXT_MINER.df.keys()), substring='_similarity')) > 0)

    def test_tfifd(self):
        _tfidf_feature: int = len(TEXT_MINER.get_str_match(cases=list(TEXT_MINER.df.keys()), substring='_tfidf'))
        TEXT_MINER.tfidf()
        self.assertTrue(expr=_tfidf_feature == 0 and len(TEXT_MINER.get_str_match(cases=list(TEXT_MINER.df.keys()), substring='_tfidf')) > 0)

    def test_translate(self):
        self.assertEqual(first='ciao', second=TEXT_MINER.translate(text='hallo', lang='it').lower())


if __name__ == '__main__':
    unittest.main()
