import copy
import emoji
import numpy as np
import pandas as pd
import re
import spacy
import subprocess

from .unsupervised_machine_learning import UnsupervisedML
from .utils import EasyExploreUtils, Log, SPECIAL_CHARACTERS, SPECIAL_SEPARATORS
from googletrans import Translator
from multiprocessing.pool import ThreadPool
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from typing import Dict, List


TEXT_FEATURE_SEGMENTS: Dict[str, List[str]] = dict(enumeration=[],
                                                   phrases=[],
                                                   id=[],
                                                   email=[],
                                                   rating=[],
                                                   url=[],
                                                   unknown=[]
                                                   )
LANG_MODELS: Dict[str, dict] = dict(#za=dict(name='chinese',
                                    #        model=dict(spacy=dict(sm='za_core_web_sm',
                                    #                              md='za_core_web_md',
                                    #                              lg='za_core_web_lg'
                                    #                              )
                                    #                   )
                                    #        ),
                                    da=dict(name='danish',
                                            model=dict(spacy=dict(sm='da_core_news_sm',
                                                                  md='da_core_news_md',
                                                                  lg='da_core_news_lg'
                                                                  )
                                                       )
                                            ),
                                    en=dict(name='english',
                                            model=dict(spacy=dict(sm='en_core_web_sm',
                                                                  md='en_core_web_md',
                                                                  lg='en_core_web_lg'
                                                                  )
                                                       )
                                            ),
                                    fr=dict(name='french',
                                            model=dict(spacy=dict(sm='fr_core_news_sm',
                                                                  md='fr_core_news_md',
                                                                  lg='fr_core_news_lg'
                                                                  )
                                                       )
                                            ),
                                    nl=dict(name='dutch',
                                            model=dict(spacy=dict(sm='nl_core_news_sm',
                                                                  md='nl_core_news_md',
                                                                  lg='nl_core_news_lg'
                                                                  )
                                                       )
                                            ),
                                    gr=dict(name='greek',
                                            model=dict(spacy=dict(sm='gr_core_news_sm',
                                                                  md='gr_core_news_md',
                                                                  lg='gr_core_news_lg'
                                                                  )
                                                       )
                                            ),
                                    de=dict(name='german',
                                            model=dict(spacy=dict(sm='de_core_news_sm',
                                                                  md='de_core_news_md',
                                                                  lg='de_core_news_lg'
                                                                  )
                                                       )
                                            ),
                                    it=dict(name='italian',
                                            model=dict(spacy=dict(sm='it_core_news_sm',
                                                                  md='it_core_news_md',
                                                                  lg='it_core_news_lg'
                                                                  )
                                                       )
                                            ),
                                    ja=dict(name='japanese',
                                            model=dict(spacy=dict(sm='ja_core_news_sm',
                                                                  md='ja_core_news_md',
                                                                  lg='ja_core_news_lg'
                                                                  )
                                                       )
                                            ),
                                    lt=dict(name='lithuanian',
                                            model=dict(spacy=dict(sm='lt_core_news_sm',
                                                                  md='lt_core_news_md',
                                                                  lg='lt_core_news_lg'
                                                                  )
                                                       )
                                            ),
                                    xx=dict(name='multi-language',
                                            model=dict(spacy=dict(sm='xx_ent_wiki_sm',
                                                                  md='xx_ent_wiki_md',
                                                                  lg='xx_ent_wiki_lg'
                                                                  )
                                                       )
                                            ),
                                    nb=dict(name='norwegian_bokmal',
                                            model=dict(spacy=dict(sm='nb_core_news_sm',
                                                                  md='nb_core_news_md',
                                                                  lg='nb_core_news_lg'
                                                                  )
                                                       )
                                            ),
                                    #pl=dict(name='polish',
                                    #        model=dict(spacy=dict(sm='pl_core_news_sm',
                                    #                              md='pl_core_news_md',
                                    #                              lg='pl_core_news_lg'
                                    #                              )
                                    #                   )
                                    #        ),
                                    pt=dict(name='portuguese',
                                            model=dict(spacy=dict(sm='pt_core_news_sm',
                                                                  md='pt_core_news_md',
                                                                  lg='pt_core_news_lg'
                                                                  )
                                                       )
                                            ),
                                    ro=dict(name='romanian',
                                            model=dict(spacy=dict(sm='ro_core_news_sm',
                                                                  md='ro_core_news_md',
                                                                  lg='ro_core_news_lg'
                                                                  )
                                                       )
                                            ),
                                    ru=dict(name='russian',
                                            model=dict(spacy=dict(sm='ru_core_news_sm',
                                                                  md='ru_core_news_md',
                                                                  lg='ru_core_news_lg'
                                                                  )
                                                       )
                                            ),
                                    es=dict(name='spanish',
                                            model=dict(spacy=dict(sm='es_core_news_sm',
                                                                  md='es_core_news_md',
                                                                  lg='es_core_news_lg'
                                                                  )
                                                       )
                                            )
                                    )
WEB_ELEMENTS: Dict[str, List[str]] = dict(protocols=['http://', 'https://', 'ftp://'],
                                          domain_ext=['org', 'com', 'onion']
                                          )

# TODO:
#  Linguistic features:
#  - derive tense from comparison between word and lemmatized version of it
#  - aspect: grammatical category which reflects the action given by the verb happened in respect to time
#  - mood: indicating whether a verb expresses a fact (indicative) or conditionality (subjunctive)
#           -> semantic notation: modality (opinion) / illocation (sentence type)
#           -> modality


class TextMinerException(Exception):
    """
    Class for setting up exceptions for class TextMiner
    """
    pass


class TextMiner:
    """
    Class for processing text data
    """
    def __init__(self,
                 df: pd.DataFrame,
                 features: List[str] = None,
                 lang: str = None,
                 lang_model: str = None,
                 lang_model_size: str = 'sm',
                 lang_model_framework: str = 'spacy',
                 segmentation_threshold: float = 0.5,
                 auto_interpret_natural_language: bool = False,
                 multi_threading: bool = True
                 ):
        """
        :param df: Pandas DataFrame
            Data set containing text or id features

        :param features: List[str]
            Name of the text or id features

        :param lang: str
            Language of the text (use multi-language framework if lang is None)

        :param lang_model: str
            Name of the language model to use

        :param lang_model_size: str
            Name of the model size type:
                -> sm, small -> small (pre-trained) language model
                -> lg, large, big -> large (pre-trained) language model

        :param lang_model_framework: str
            Name of the language model framework

        :param segmentation_threshold: float
            Threshold for identify certain segments

        :param auto_interpret_natural_language: bool
            Whether to interpret natural language automatically while initialization

        :param multi_threading: bool
            Whether to run text processing using multiple threads or just a single thread
        """
        Log(write=False, logger_file_path=None).log(msg='Initialize text miner ...')
        if df.shape[0] == 0:
            raise TextMinerException('No cases found in data set')
        if df.shape[1] == 0:
            raise TextMinerException('No features found in data set')
        if features is None:
            self.features: List[str] = [text_feature for text_feature in df.keys() if str(df[text_feature].dtype).find('object') >= 0]
        else:
            if len(features) > 0:
                self.features: List[str] = [text_feature for text_feature in features if str(df[text_feature].dtype).find('object') >= 0]
            if len(self.features) == 0:
                self.features: List[str] = [text_feature for text_feature in df.keys() if str(df[text_feature].dtype).find('object') >= 0]
        if len(self.features) == 0:
            raise TextMinerException('No text feature found in data set')
        self.df: pd.DataFrame = df[self.features]
        self.text_feature: str = ''
        self.lang: str = lang if lang != 'xx' else None
        if self.lang is not None and self.lang not in LANG_MODELS.keys():
            for l, language in enumerate(LANG_MODELS.keys()):
                if self.lang.lower() == LANG_MODELS[language].get('name'):
                    self.lang = language
                    break
                if l + 1 == len(LANG_MODELS.keys()):
                    Log(write=False, level='info').log(msg='Language ({}) not supported. Use auto-detection instead'.format(self.lang))
        self.detected_language: dict = {}
        if lang_model_size in ['sm', 'small']:
            self.lang_model_size: str = 'sm'
        elif lang_model_size in ['md', 'mid']:
            self.lang_model_size: str = 'md'
        elif lang_model_size in ['lg', 'large', 'big']:
            self.lang_model_size: str = 'lg'
        else:
            self.lang_model_size: str = 'sm'
        self.lang_model_name: str = lang_model
        self.lang_model_framework: str = lang_model_framework
        self.lang_models: dict = dict(spacy={}, bert={}, roberta={}, xlnet={}, albert={})
        self.translator: Translator = Translator()
        self.ner: dict = {}
        self.pos: dict = {}
        self.dep: dict = {}
        self.web: dict = {}
        self.emoji: dict = {}
        self.numbers: dict = {}
        self.stop_words: dict = {}
        self.special_chars: dict = {}
        self.clean_phrases: dict = {}
        self.cluster: dict = {}
        self.similarity_scores: dict = {}
        self.enumeration: dict = {}
        self.internal_separator: str = '||'
        self.multi_threading: bool = multi_threading
        self.auto_interpretation: bool = auto_interpret_natural_language
        self.segments: Dict[str, List[str]] = TEXT_FEATURE_SEGMENTS
        self.segment_threshold: float = segmentation_threshold if (segmentation_threshold > 0) and (segmentation_threshold <= 1) else 0.5
        Log(write=False, logger_file_path=None).log(msg='Start segmentation of (text) features ...')
        self._segmentation(enumeration=True, phrases=True, web=True, identifier=True)
        if self.auto_interpretation:
            if len(self.segments.get('phrases')) > 0:
                Log(write=False, logger_file_path=None).log(msg='Recognized natural language. Start interpretation ...')
                self._interpret_text()
        Log(write=False, logger_file_path=None).log(msg='Text {} finished. Happy Mining :)'.format('processing' if self.auto_interpretation else 'segmentation'))

    def _apply_clustering(self, features: List[str], k: int = 3):
        """
        Apply text clustering

        :param features: List[str]
            Name of the features to cluster text content

        :param k: int
            Number of clusters to generate
        """
        for feature in features:
            self.cluster.update({feature: {}})
            self.df[feature] = self.df[feature].fillna('')
            _tfidf_vectorizer: TfidfVectorizer = TfidfVectorizer(stop_words=LANG_MODELS.get(self.lang)['name'],
                                                                 ngram_range=(1, 3)
                                                                 )
            _clean_text_data: pd.DataFrame = self.df[feature].apply(lambda x: self._clean_text(phrase=str(x)))
            _tfidf_matrix: pd.DataFrame = pd.DataFrame(data=_tfidf_vectorizer.fit_transform(_clean_text_data).data,
                                                       columns=[feature]
                                                       )
            _text_clustering: dict = UnsupervisedML(df=_tfidf_matrix,
                                                    cluster_algorithms=['kmeans'],
                                                    n_cluster_components=3
                                                    ).ml_pipeline()

    def _apply_similarity(self, features: List[str]):
        """
        Apply similarity scoring

        :param features: List[str]
            Name of the features to calculate similarity of text feature
        """
        for feature in features:
            self.similarity_scores.update({feature: {}})
            for x, case in enumerate(self.df[feature].values.tolist()):
                self.similarity_scores[feature].update({x: []})
                if isinstance(case, str):
                    if feature in self.detected_language.keys():
                        _lang: str = self.detected_language[feature]['most_freq']
                    else:
                        _lang: str = self.lang
                    _model = self.lang_models[self.lang_model_framework][_lang]['model']
                    _nlp = _model(self._clean_text(phrase=str(case)))
                    for other_case in self.df[feature].values.tolist():
                        if case != other_case:
                            self.similarity_scores[feature][x].append(self.lang_models[self.lang_model_framework][_lang]['model'](self._clean_text(phrase=other_case)))
                elif isinstance(case, float):
                    for other_case in self.df[feature].values.tolist():
                        if case != other_case:
                            self.similarity_scores[feature][x].append(0)

    def _clean_text(self,
                    phrase: str,
                    numbers: bool = True,
                    stop_words: bool = True,
                    special_characters: bool = True,
                    punct: bool = True,
                    pron: bool = True,
                    entity: bool = True,
                    lemmatizing: bool = True
                    ) -> str:
        """
        Clean phrase from stop-words, punctuations and pronouns

        :param phrase: str
            Text phrase to clean

        :param numbers: bool
            Whether to remove numbers from text or not

        :param stop_words: bool
            Whether to remove stop-words from text or not

        :param special_characters: bool
            Whether to remove special characters from text or not

        :param punct: bool
            Whether to remove punctuation from text or not

        :param pron: bool
            Whether to remove pronouns from text or not

        :param entity: bool
            Whether to remove recognized entities from text or not

        :param lemmatizing: bool
            Whether to lemmatize (trim words to their word-stem) text or not

        :return: str:
            Cleaned text phrase
        """
        _phrase = []
        _lang: str = self.translator.detect(text=phrase).lang
        if _lang in self.lang_models[self.lang_model_framework].keys():
            _model = self.lang_models[self.lang_model_framework][_lang]['model']
            _nlp = _model(phrase.lower())
            for token in _nlp:
                if (numbers and token.is_digit) or (numbers and token.like_num):
                    continue
                if stop_words and token.is_stop:
                    continue
                if special_characters and token.text in SPECIAL_CHARACTERS:
                    continue
                if punct and token.is_punct:
                    continue
                if pron and token.lemma_ == '-PRON-':
                    continue
                if entity and token.ent_type > 0:
                    continue
                if lemmatizing:
                    _phrase.append(token.lemma_)
                else:
                    _phrase.append(token)
            if len(_phrase) == 0:
                return ''
            return ' '.join(_phrase)
        return phrase

    def _extract_email(self, feature: str = None):
        """
        Extract email address from text

        :param feature: str
            Name of the text feature
        """
        if feature is not None and feature in self.segments.get('email'):
            _features: List[str] = [feature]
        else:
            _features: List[str] = self.segments.get('email')
        for feature in _features:
            self.df['email'] = self.df[feature].apply(lambda x: x.split('@')[1] if x.find('@') >= 0 else np.nan)

    @staticmethod
    def _extract_emojis(text: str) -> list:
        """
        Extract emojis from text

        :param text: str
            Text data

        :return: list
            Extracted emojis
        """
        return [e for e in str(text) if e in emoji.UNICODE_EMOJI]

    def _extract_url(self, feature: str = None):
        """
        Extract parts of url

        :param feature: str
            Name of the feature to process
        """
        if feature is not None and feature in self.segments.get('url'):
            _features: List[str] = [feature]
        else:
            _features: List[str] = self.segments.get('url')
        for feature in _features:
            self.df['domain'] = self.df[feature].apply(lambda x: x.split('.')[1] if x.find('www.') >= 0 else x.split('.')[0])
            self.df['domain_ext'] = self.df[feature].apply(lambda x: x.split('.')[-1].split('/')[0] if x.find('www.') >= 0 else x.split('.')[-1].split('/')[0])

    def _get_lang(self):
        """
        Get language observed directly from text input
        """
        _text_samples: List[str] = []
        _lang_each_case: List[str] = []
        for _ in range(0, 10, 1):
            _text_samples.append(np.random.choice(a=self.df.loc[~self.df[self.text_feature].isnull(), self.text_feature].values.tolist()))
            _lang_each_case.append(self.translator.detect(text=str(_text_samples[-1])).lang)
        _lang_each_case = list(set(_lang_each_case))
        for lang in copy.deepcopy(_lang_each_case):
            if lang not in LANG_MODELS.keys():
                del _lang_each_case[_lang_each_case.index(lang)]
                Log(write=False, level='info').log(msg='Language ({}) not supported'.format(lang))
        if len(_lang_each_case) > 0:
            self.detected_language.update({self.text_feature: dict(unique=_lang_each_case,
                                                                   most_freq=EasyExploreUtils().get_freq(data=_lang_each_case).most_common(1)[0][0]
                                                                   )
                                           })

    def _get_lang_model(self):
        """
        Load language model
        """
        if self.lang not in LANG_MODELS.keys():
            self._get_lang()
            if self.detected_language.get(self.text_feature) is None:
                raise TextMinerException('No supported language found')
            self.lang = self.detected_language[self.text_feature]['most_freq']
        _language_model: str = LANG_MODELS[self.lang]['model'][self.lang_model_framework][self.lang_model_size]
        if self.lang_model_framework == 'spacy':
            try:
                _spacy = spacy.load(_language_model)
            except OSError:
                try:
                    subprocess.run('python -m spacy download {}'.format(_language_model), shell=True)
                except IOError:
                    subprocess.run('python -m pip install spacy', shell=True)
                finally:
                    subprocess.run('python -m spacy download {}'.format(_language_model), shell=True)
            finally:
                _spacy = spacy.load(_language_model)
            self.lang_models[self.lang_model_framework].update({self.lang: dict(name=_language_model, model=copy.deepcopy(_spacy))})

    def _get_stop_words(self, lang: str) -> List[str]:
        """
        Get list of stop words language based

        :param lang: str
            Language

        :return: List[str]:
             Language based stop words
        """
        if lang == 'xx':
            _multi_lang_stop_words: List[str] = []
            for supported_lang in ['nl', 'en', 'fr', 'de', 'it', 'pt', 'ru', 'es']:
                try:
                    _multi_lang_stop_words.extend(stopwords.words(LANG_MODELS[supported_lang]['name']))
                except AttributeError:
                    try:
                        subprocess.run('python -m spacy download {}'.format(LANG_MODELS[supported_lang]['model']['spacy'][self.lang_model_size]), shell=True)
                    except IOError:
                        subprocess.run('python -m pip install spacy', shell=True)
                    finally:
                        subprocess.run('python -m spacy download {}'.format(LANG_MODELS[supported_lang]['model']['spacy'][self.lang_model_size]), shell=True)
                finally:
                    _multi_lang_stop_words.extend(stopwords.words(LANG_MODELS[supported_lang]['name']))
            return _multi_lang_stop_words
        else:
            try:
                return stopwords.words(LANG_MODELS[lang]['name'])
            except OSError:
                Log(write=False, level='info').log(msg='Language ({}) not supported'.format(lang))
                return []

    def _interpret_text(self):
        """
        Interpret text
        """
        _thread_pool: ThreadPool = ThreadPool(processes=len(self.segments.get('phrases')))
        for feature in self.segments.get('phrases'):
            self.special_chars.update({feature: dict(count=[], count_unique=[])})
            self.clean_phrases.update({feature: dict(clean=[],
                                                     count_len=[],
                                                     count_tokens=[],
                                                     count_words=[],
                                                     count_sents=[],
                                                     count_conjuncts=[],
                                                     count_url=[],
                                                     count_email=[]
                                                     )
                                       })
            self.emoji.update({feature: dict(has=[], count=[], count_unique=[], label=[])})
            self.pos.update({feature: dict(count_unique_tags=[],
                                           count_unique_labels=[],
                                           count_ad=[],
                                           count_verb=[],
                                           count_noun=[],
                                           pos_seq=[]
                                           )
                             })
            self.ner.update({feature: dict(recognized=[], count=[], count_unique=[], entity=[], entity_type=[])})
            self.dep.update({feature: dict(noun_pairs=dict(count_dep=[],
                                                           count_unique_dep=[],
                                                           count_root_dep=[],
                                                           count_unique_root_dep=[],
                                                           count_root_head_dep=[],
                                                           count_unique_root_head_dep=[],
                                                           count_root_text_dep=[],
                                                           count_unique_root_text_dep=[]
                                                           ),
                                           tree=dict(count_has_child=[],
                                                     count_dep_of_child=[],
                                                     count_unique_dep_of_child=[],
                                                     count_head_text_of_child=[],
                                                     count_unique_head_text_of_child=[],
                                                     count_head_pos_of_child=[],
                                                     count_unique_head_pos_of_child=[],
                                                     count_children=[],
                                                     children=[]
                                                     )
                                           )
                             })
            if self.multi_threading:
                _thread_pool.apply(func=self._process_text, kwds=dict(feature=feature,
                                                                      decap=True,
                                                                      numbers=True,
                                                                      stop_words=True,
                                                                      punct=True,
                                                                      pron=True,
                                                                      entity=True,
                                                                      lemmatizing=True,
                                                                      handle_emojis='replace'
                                                                      )
                                   )
            else:
                self._process_text(feature=feature,
                                   decap=True,
                                   numbers=True,
                                   stop_words=True,
                                   punct=True,
                                   pron=True,
                                   entity=True,
                                   lemmatizing=True,
                                   handle_emojis='replace'
                                   )
            Log(write=False, level='info').log(msg='Feature "{}" interpreted'.format(feature))

    def _is_enumeration(self) -> bool:
        """
        Check whether feature contains enumerated information or not

        :return dict:
            Whether feature contains enumeration or not
        """
        for char in SPECIAL_SEPARATORS:
            if all(self.df.loc[~self.df[self.text_feature].isnull(), self.text_feature].str.find(char)) >= 0:
                _df: pd.DataFrame = copy.deepcopy(self.df[self.text_feature])
                _df['sep_len'] = _df.apply(lambda x: len(x.split(char)) if x == x else 0)
                _split_cases: int = len(_df['sep_len'][(_df['sep_len'] > 1)].values)
                if _split_cases >= (self.df.shape[0] * 0.05):
                    self.enumeration.update({self.text_feature: char})
                    return True
        return False

    def _is_email_address(self) -> bool:
        """
        Check whether feature contains email addresses or not

        :return bool:
            Whether feature contains email addresses or not
        """
        _signals: int = 0
        _df: pd.DataFrame = copy.deepcopy(self.df.loc[~self.df[self.text_feature].isnull(), self.text_feature])
        if any(_df.str.find(' ')) >= 0:
            return False
        if any(_df.str.find('@')) >= 0:
            _signals += 1
            if any(_df.str.find('.')) >= 0:
                _signals += 1
                for ext in WEB_ELEMENTS.get('domain_ext') + list(LANG_MODELS.keys()):
                    if any(_df.str.split('.')[-1]) == ext:
                        _signals += 1
                        break
        if _signals > 2:
            return True
        return False

    def _is_geo(self) -> bool:
        """
        Check whether feature can be interpreted as geo feature
        """
        return False

    def _is_phrase(self) -> bool:
        """
        Check whether feature contains phrases or not

        :return bool:
            Whether feature contains phrases or not
        """
        _signals: int = 0
        _df: pd.DataFrame = copy.deepcopy(self.df.loc[~self.df[self.text_feature].isnull(), self.text_feature])
        if _df.shape[0] == _df.loc[_df.str.find(' ') < 0].shape[0]:
            return False
        for char in SPECIAL_CHARACTERS:
            if char != ' ':
                if _df.loc[_df.str.find(char) >= 0].shape[0] > 0:
                    _signals += 1
                    break
        try:
            self._get_lang_model()
        except OSError:
            self._get_lang_model()
        except TextMinerException:
            return False
        _model = self.lang_models[self.lang_model_framework][self.lang]['model']
        _part_of_speech_signals: int = 0
        _found_stop_word: bool = False
        for text in _df:
            _nlp = _model(text)
            for token in _nlp:
                if token.text in self._get_stop_words(lang=self.lang):
                    if not _found_stop_word:
                        _found_stop_word = True
                        _signals += 1
                        _part_of_speech_signals += 1
                if token.pos_ in ['VERB', 'ADV', 'ADJ', 'ADP']:
                    _signals += 2
                    _part_of_speech_signals += 1
            if _part_of_speech_signals > 0:
                break
        if _signals > 2:
            return True
        return False

    def _is_rating(self) -> bool:
        """
        Check whether feature contains ratings only
        """
        pass

    def _is_url(self) -> bool:
        """
        Check whether feature contains url or not

        :return bool:
            Whether feature contains url or not
        """
        _signals: int = 0
        _df: pd.DataFrame = copy.deepcopy(self.df.loc[~self.df[self.text_feature].isnull(), self.text_feature])
        if _df.loc[_df.str.find(' ') >= 0].shape[0] > 0:
            if _df.shape[0] == _df.loc[_df.str.find(' ') >= 0].shape[0]:
                return False
        for protocol in WEB_ELEMENTS.get('protocols'):
            if _df.loc[_df.str.find(protocol) >= 0].shape[0] > 0:
                if _df.shape[0] == _df.loc[_df.str.find(protocol) >= 0].shape[0]:
                    return True
                else:
                    if (_df.loc[_df.str.find(protocol) >= 0].shape[0] / _df.shape[0]) >= self.segment_threshold:
                        _signals += 1
        if _df.loc[_df.str.find('www.') >= 0].shape[0] > 0:
            if _df.shape[0] == _df.loc[_df.str.find('www.') >= 0].shape[0]:
                return True
            else:
                if (_df.shape[0] == _df.loc[_df.str.find('www.') >= 0].shape[0] / _df.shape[0]) >= self.segment_threshold:
                    _signals += 1
        if _signals == 0:
            for ext in WEB_ELEMENTS.get('domain_ext'):
                if _df.loc[_df.str.find('.{}'.format(ext)) >= 0].shape[0] > 0:
                    if _df.shape[0] == _df.loc[_df.str.find('.{}'.format(ext)) >= 0].shape[0]:
                        return True
                    else:
                        if (_df.shape[0] == _df.loc[_df.str.find('.{}'.format(ext)) >= 0].shape[0] / _df.shape[0]) >= self.segment_threshold:
                            _signals += 1
                else:
                    for other_ext in LANG_MODELS.keys():
                        if _df.loc[_df.str.find('.{}'.format(other_ext)) >= 0].shape[0] > 0:
                            if _df.shape[0] == _df.loc[_df.str.find('.{}'.format(other_ext)) >= 0].shape[0]:
                                return True
                            else:
                                if (_df.shape[0] == _df.loc[_df.str.find('.{}'.format(other_ext)) >= 0].shape[0] / _df.shape[0]) >= self.segment_threshold:
                                    _signals += 1
        if _signals > 0:
            return True
        return False

    def _process_text(self,
                      feature: str,
                      decap: bool = True,
                      numbers: bool = True,
                      stop_words: bool = True,
                      special_chars: bool = True,
                      punct: bool = True,
                      pron: bool = True,
                      web: bool = True,
                      entity: bool = True,
                      lemmatizing: bool = True,
                      handle_emojis: str = 'replace'
                      ):
        """
        Process text

        :param feature: str
            Name of the text feature to process

        :param decap: bool
            Whether to decapitulate text or not

        :param numbers: bool
            Whether to remove numbers from text or not

        :param stop_words: bool
            Whether to remove stop-words from text or not

        :param special_chars: bool
            Whether to remove all special characters from text or not

        :param punct: bool
            Whether to remove punctuation from text or not

        :param pron: bool
            Whether to remove pronouns from text or not

        :param web: bool
            Whether to remove all web elements like url or email from text or not

        :param entity: bool
            Whether to remove recognized entities from text or not

        :param lemmatizing: bool
            Whether to lemmatize (trim words to their word-stem) text or not

        :param handle_emojis: str
            Handle emojis properly:
                -> replace: Replace emoji unicode character with pre-defined label
                -> clean, remove, erase, delete: Remove emoji from text
                -> ignore: Ignore emojis
        """
        for x, case in enumerate(self.df[feature].values):
            _case: str = str(case).lower() if decap else case
            _emoji_handler: str = handle_emojis
            if handle_emojis not in ['clean', 'remove', 'erase', 'delete', 'replace', 'ignore']:
                _emoji_handler: str = 'ignore'
            if isinstance(case, str):
                _special_chars: List[str] = []
                _pos: dict = dict(token=[], pos=[], tag=[], ad=[], verb=[], noun=[])
                _noun_chunks: dict = dict(pair=[], root=[], root_dep=[], root_head=[])
                _tree: dict = dict(dep=[], head=[], head_pos=[], has_children=[], children=[], childrens=0)
                _text: dict = dict(len=0, tokens=0, words=[], sents=[], conjuncts=[], url=[], email=[])
                _model = self.lang_models[self.lang_model_framework][self.lang]['model']
                _nlp = _model(_case)
                for sent in _nlp.sents:
                    _text['len'] += len(sent.text)
                    _text['tokens'] += len(sent.text.split(' '))
                    _text['sents'].append(sent.text)
                    _text['conjuncts'].extend(list(sent.conjuncts))
                for chunk in _nlp.noun_chunks:
                    _noun_chunks['pair'].append(chunk.text)
                    _noun_chunks['root'].append(chunk.root.text)
                    _noun_chunks['root_dep'].append(chunk.root.dep_)
                    _noun_chunks['root_head'].append(chunk.root.head.text)
                self.dep[feature]['noun_pairs']['count_dep'].append(len(_noun_chunks['pair']))
                self.dep[feature]['noun_pairs']['count_unique_dep'].append(len(list(set(_noun_chunks['pair']))))
                self.dep[feature]['noun_pairs']['count_root_dep'].append(len(_noun_chunks['root']))
                self.dep[feature]['noun_pairs']['count_unique_root_dep'].append(len(list(set(_noun_chunks['root']))))
                self.dep[feature]['noun_pairs']['count_root_head_dep'].append(len(_noun_chunks['root_dep']))
                self.dep[feature]['noun_pairs']['count_unique_root_head_dep'].append(len(list(set(_noun_chunks['root_dep']))))
                self.dep[feature]['noun_pairs']['count_root_text_dep'].append(len(_noun_chunks['root_head']))
                self.dep[feature]['noun_pairs']['count_unique_root_text_dep'].append(len(list(set(_noun_chunks['root_head']))))
                _phrase: List[str] = []
                for token in _nlp:
                    if _emoji_handler != 'ignore' and token in emoji.UNICODE_EMOJI:
                        if len(self.emoji[feature].get('has')) == 0:
                            self.emoji[feature]['has'].append(1)
                            self.emoji[feature]['count'].append(len(re.findall(pattern=r'[^\w\s,]', string=_case)))
                            self.emoji[feature]['count_unique'].append(len(pd.unique(re.findall(pattern=r'[^\w\s,]', string=_case))))
                            self.emoji[feature]['label'].append(emoji.demojize(string=token, use_aliases=False))
                        if _emoji_handler == 'replace':
                            _phrase.append(emoji.demojize(string=token, use_aliases=False))
                        continue
                    _pos['pos'].append(token.pos_)
                    _pos['tag'].append(token.tag_)
                    _pos['token'].append(token.text)
                    _tree['dep'].append(token.dep_)
                    _tree['head'].append(token.head.text)
                    _tree['head_pos'].append(token.head.pos_)
                    _children: List[str] = [str(child) for child in token.children]
                    _tree['has_children'].append(1 if len(_children) > 0 else 0)
                    _tree['childrens'] += len(_children)
                    _tree['children'].append(self.internal_separator.join(_children) if len(_children) > 0 else np.nan)
                    if token.pos_ in ['ADJ', 'ADV']:
                        _pos['ad'].append(token.text)
                    if token.pos_ == 'VERB':
                        _pos['verb'].append(token.text)
                    if token.pos_ == 'NOUN':
                        _pos['noun'].append(token.text)
                    if (numbers and token.is_digit) or (numbers and token.like_num):
                        continue
                    if stop_words and token.is_stop:
                        _text['words'].append(token.text)
                        continue
                    if special_chars and token.text in SPECIAL_CHARACTERS:
                        _special_chars.append(token.text)
                        continue
                    if punct and token.is_punct:
                        continue
                    if pron and token.lemma_ == '-PRON-':
                        continue
                    if (web and token.like_url) or (web and token.like_email):
                        if token.like_url:
                            _text['url'].append(token.text)
                        if token.like_email:
                            _text['email'].append(token.text)
                        continue
                    if entity and token.ent_type > 0:
                        continue
                    if lemmatizing:
                        _phrase.append(token.lemma_)
                    else:
                        _phrase.append(token.text)
                    _text['words'].append(token.text)
                if len(_phrase) == 0:
                    self.clean_phrases[feature]['clean'].append('')
                else:
                    self.clean_phrases[feature]['clean'].append(copy.deepcopy(' '.join(_phrase)))
                self.special_chars[feature]['count'].append(copy.deepcopy(len(_special_chars)))
                self.special_chars[feature]['count_unique'].append(copy.deepcopy(len(list(set(_special_chars)))))
                self.clean_phrases[feature]['count_len'].append(copy.deepcopy(_text.get('len')))
                self.clean_phrases[feature]['count_tokens'].append(copy.deepcopy(_text.get('token')))
                self.clean_phrases[feature]['count_words'].append(copy.deepcopy(len(_text.get('words'))))
                self.clean_phrases[feature]['count_sents'].append(copy.deepcopy(len(_text.get('sents'))))
                self.clean_phrases[feature]['count_conjuncts'].append(copy.deepcopy(len(_text.get('conjuncts'))))
                self.clean_phrases[feature]['count_url'].append(copy.deepcopy(len(_text.get('url'))))
                self.clean_phrases[feature]['count_email'].append(copy.deepcopy(len(_text.get('email'))))
                self.pos[feature]['count_unique_tags'].append(len(list(set(_pos['tag']))))
                self.pos[feature]['count_unique_labels'].append(len(list(set(_pos['pos']))))
                self.pos[feature]['count_ad'].append(len(list(set(_pos['ad']))))
                self.pos[feature]['count_verb'].append(len(list(set(_pos['verb']))))
                self.pos[feature]['count_noun'].append(len(list(set(_pos['noun']))))
                self.dep[feature]['tree']['count_has_child'].append(len(_tree['has_children']))
                self.dep[feature]['tree']['count_dep_of_child'].append(len(_tree['dep']))
                self.dep[feature]['tree']['count_unique_dep_of_child'].append(len(list(set(_tree['dep']))))
                self.dep[feature]['tree']['count_head_text_of_child'].append(len(_tree['head']))
                self.dep[feature]['tree']['count_unique_head_text_of_child'].append(len(list(set(_tree['head']))))
                self.dep[feature]['tree']['count_head_pos_of_child'].append(len(_tree['head_pos']))
                self.dep[feature]['tree']['count_unique_head_pos_of_child'].append(len(list(set(_tree['head_pos']))))
                self.dep[feature]['tree']['count_children'].append(_tree['childrens'])
                self.dep[feature]['tree']['children'].append(_tree['children'])
                if len(_nlp.ents) > 0:
                    _entity: List[str] = [ent.text for ent in _nlp.ents]
                    _entity_type: List[str] = [ent.label_ for ent in _nlp.ents]
                    self.ner[feature]['recognized'].append(1 if len(_entity) > 0 else 0)
                    self.ner[feature]['count'].append(len(_entity))
                    self.ner[feature]['count_unique'].append(len(list(set(_entity))))
                    self.ner[feature]['entity'].append(self.internal_separator.join(_entity) if len(_entity) > 0 else np.nan)
                    self.ner[feature]['entity_type'].append(self.internal_separator.join(_entity_type) if len(_entity_type) > 0 else np.nan)
                    if len(_entity) > 0:
                        self.enumeration.update({'{}_ner_entity'.format(feature): self.internal_separator,
                                                 '{}_ner_entity_type'.format(feature): self.internal_separator
                                                 })
                else:
                    self.ner[feature]['recognized'].append(0)
                    self.ner[feature]['count'].append(0)
                    self.ner[feature]['count_unique'].append(0)
                    self.ner[feature]['entity'].append(np.nan)
                    self.ner[feature]['entity_type'].append(np.nan)
                self.enumeration.update({'{}_dep_tree_children'.format(feature): self.internal_separator})
            else:
                self.special_chars[feature]['count'].append(0)
                self.special_chars[feature]['count_unique'].append(0)
                self.clean_phrases[feature]['count_len'].append(0)
                self.clean_phrases[feature]['count_tokens'].append(0)
                self.clean_phrases[feature]['count_words'].append(0)
                self.clean_phrases[feature]['count_sents'].append(0)
                self.clean_phrases[feature]['count_conjuncts'].append(0)
                self.emoji[feature]['has'].append(0)
                self.emoji[feature]['count'].append(0)
                self.emoji[feature]['count_unique'].append(0)
                self.emoji[feature]['label'].append(np.nan)
                self.pos[feature]['count_unique_tags'].append(0)
                self.pos[feature]['count_unique_labels'].append(0)
                self.pos[feature]['count_ad'].append(0)
                self.pos[feature]['count_verb'].append(0)
                self.pos[feature]['count_noun'].append(0)
                self.ner[feature]['recognized'].append(0)
                self.ner[feature]['count'].append(0)
                self.ner[feature]['count_unique'].append(0)
                self.ner[feature]['entity'].append(np.nan)
                self.ner[feature]['entity_type'].append(np.nan)
                self.dep[feature]['tree']['count_has_child'].append(0)
                self.dep[feature]['tree']['count_dep_of_child'].append(0)
                self.dep[feature]['tree']['count_unique_dep_of_child'].append(0)
                self.dep[feature]['tree']['count_head_text_of_child'].append(0)
                self.dep[feature]['tree']['count_unique_head_text_of_child'].append(0)
                self.dep[feature]['tree']['count_head_pos_of_child'].append(0)
                self.dep[feature]['tree']['count_unique_head_pos_of_child'].append(0)
                self.dep[feature]['tree']['count_children'].append(0)
                self.dep[feature]['tree']['children'].append(np.nan)
                self.dep[feature]['noun_pairs']['count_dep'].append(0)
                self.dep[feature]['noun_pairs']['count_unique_dep'].append(0)
                self.dep[feature]['noun_pairs']['count_root_dep'].append(0)
                self.dep[feature]['noun_pairs']['count_unique_root_dep'].append(0)
                self.dep[feature]['noun_pairs']['count_root_head_dep'].append(0)
                self.dep[feature]['noun_pairs']['count_unique_root_head_dep'].append(0)
                self.dep[feature]['noun_pairs']['count_root_text_dep'].append(0)
                self.dep[feature]['noun_pairs']['count_unique_root_text_dep'].append(0)

    def _segmentation(self,
                      enumeration: bool = True,
                      phrases: bool = True,
                      web: bool = True,
                      identifier: bool = True,
                      geo: bool = True
                      ):
        """
        Analyzing text content in order to identify important features potentially

        :param enumeration: bool
            Whether to identify enumeration or not

        :param phrases: bool
            Whether to identify phrases or not

        :param web: bool
            Whether to identify urls or email addresses or not

        :param identifier: bool
            Whether to identify id's

        :param geo: bool
            Whether to identify geo feature (like city, state, country, location, etc.)
        """
        for feature in self.features:
            self.text_feature = copy.deepcopy(feature)
            if web:
                if self._is_email_address():
                    self.segments['email'].append(feature)
                    Log(write=False, logger_file_path=None).log(msg='Recognized feature "{}" as email-address'.format(feature))
                    continue
                if self._is_url():
                    self.segments['url'].append(feature)
                    Log(write=False, logger_file_path=None).log(msg='Recognized feature "{}" as url'.format(feature))
                    continue
            if enumeration:
                if self._is_enumeration():
                    self.segments['enumeration'].append(feature)
                    Log(write=False, logger_file_path=None).log(msg='Recognized feature "{}" as enumeration'.format(feature))
                    continue
            if phrases:
                if self._is_phrase():
                    self.segments['phrases'].append(feature)
                    Log(write=False, logger_file_path=None).log(msg='Recognized feature "{}" as natural language'.format(feature))
                    continue
            if identifier:
                if self.df.shape[0] == len(self.df[feature].unique()):
                    self.segments['id'].append(feature)
                    Log(write=False, logger_file_path=None).log(msg='Recognized feature "{}" as id'.format(feature))
                    continue
            if geo:
                if self._is_geo():
                    self.segments['geo'].append(feature)
                    Log(write=False, logger_file_path=None).log(msg='Recognized feature "{}" as geo'.format(feature))
                    continue
            self.segments['unknown'].append(feature)
            Log(write=False, logger_file_path=None).log(msg='Feature "{}" cannot be recognized'.format(feature))

    def count_occurances(self,
                         features: List[str] = None,
                         search_text: str = None,
                         count_length: bool = False,
                         count_numbers: bool = False,
                         count_characters: bool = False,
                         count_special_characters: bool = False
                         ):
        """
        Count matches in text

        :param features: List[str]
            Name of the features to analyze

        :param search_text: str
            Text element to count

        :param count_length: bool
            Whether to count text length (all text elements combined) or not

        :param count_numbers: bool
            Whether to count all numbers in text or not

        :param count_characters: bool
            Whether to count all characters in text or not

        :param count_special_characters: bool
            Whether to count all special characters in text or not
        """
        _features: List[str] = self.segments.get('phrases') if features is None else features
        for feature in _features:
            if feature in self.df.keys():
                if search_text is not None:
                    if len(search_text) > 0:
                        self.df['count_{}'.format(search_text)] = self.df[feature].str.count(search_text)
                if count_length:
                    self.df['count_len'] = self.df[feature].str.len()
                if count_numbers:
                    self.df['count_numbers'] = self.df[feature].str.count(r'[0-9]')
                if count_characters:
                    self.df['count_characters'] = self.df[feature].str.count(r'[a-zA-Z]')
                if count_special_characters:
                    self.df['count_characters'] = self.df[feature].str.count(r'[^\0-9\a-zA-Z\s,]')

    def clustering(self):
        """
        Cluster content of text features using unsupervised machine learning (clustering)
        """
        raise NotImplemented('Text clustering not supported')

    def emoji_handler(self,
                      features: List[str] = None,
                      has_emojis: bool = True,
                      count_emojis: bool = True,
                      count_unique_emojis: bool = True,
                      convert_emoji_to_text: bool = True
                      ):
        """
        Process emojis into categorical (and semi-continuous) features

        :param features: List[str]
            Name of the features to analyze

        :param has_emojis: bool
            Whether to check if text contains emoji or not

        :param count_emojis: bool
            Whether to count all emojis or not

        :param count_unique_emojis: bool
             Whether to count unique emojis or not

        :param convert_emoji_to_text: bool
            Whether to convert each emoji into text or not
        """
        _features: List[str] = self.segments.get('phrases') if features is None else features
        for feature in _features:
            if feature in self.df.keys():
                _df: pd.DataFrame = self.df[feature].apply(lambda x: 1 if x in emoji.UNICODE_EMOJI else 0)
                if has_emojis:
                    self.df['has_emojis'] = self.df[feature].apply(lambda x: 1 if x in emoji.UNICODE_EMOJI else 0)
                if count_emojis:
                    self.df['count_emojis'] = self.df[feature].apply(lambda x: len(self._extract_emojis(text=x)))
                if count_unique_emojis:
                    self.df['count_unique_emojis'] = self.df[feature].apply(lambda x: len(pd.unique(self._extract_emojis(text=x))))
                if convert_emoji_to_text:
                    self.df[feature] = self.df[feature].apply(lambda x: emoji.demojize(string=str(x), use_aliases=False))

    def get_generated_features(self) -> pd.DataFrame:
        """
        Get generated features

        :return: Pandas DataFrame
            Data set containing generated features only
        """
        _generated_features: List[str] = []
        for segment in self.segments.keys():
            if segment != 'unknown':
                for feature in self.segments.get(segment):
                    _features: List[str] = self.get_str_match(cases=list(self.df.keys()), substring='{}_'.format(feature))
                    if len(_features) > 0:
                        _generated_features.extend(_features)
        if len(_generated_features) == 0:
            Log(write=False, level='info').log(msg='No generated features found')
            return pd.DataFrame()
        else:
            return self.df[_generated_features]

    @staticmethod
    def get_str_match(cases: List[str], substring: str) -> List[str]:
        """
        Get all matches of substring in given list

        :param cases: List[str]
            Cases to check match

        :param substring: str
            String to match

        :return: List[str]
            Matches (substring at least)
        """
        return [case for case in cases if str(case).find(substring) >= 0]

    def detect_lang(self, features: List[str] = None, sampling: bool = True):
        """
        Detect language of text phrase

        :param features: List[str]
            Name of the features to use

        :param sampling: bool
            Whether to draw sample for detecting language of text features (to avoid request error) or not
        """
        _features: List[str] = self.segments.get('phrases') if features is None else features
        for feature in _features:
            if feature in self.df.keys():
                if sampling:
                    self.text_feature = copy.deepcopy(feature)
                    self._get_lang()
                    self.df['{}_lang'.format(feature)] = self.detected_language.get(feature)
                else:
                    self.df['{}_lang'.format(feature)] = self.df[feature].apply(lambda x: self.translator.detect(text=x).lang)
            else:
                Log(write=False, level='info').log(msg='Feature "{}" not found in data set'.format(feature))

    def generate_linguistic_features(self, features: List[str] = None):
        """
        Generate numerical features based on linguistic analysis of text features

        :param features: List[str]
            Name of the features to generate linguistic features from
        """
        _features: List[str] = self.segments.get('phrases') if features is None else features
        for feature in _features:
            if feature in self.segments.get('phrases'):
                if feature not in self.clean_phrases.keys():
                    self._process_text(feature=feature,
                                       decap=True,
                                       numbers=False,
                                       stop_words=True,
                                       special_chars=True,
                                       punct=True,
                                       pron=True,
                                       web=True,
                                       entity=True,
                                       lemmatizing=True,
                                       handle_emojis='ignore'
                                       )
                self.df['{}_count_special_chars'.format(feature)] = self.special_chars[feature]['count']
                self.df['{}_count_unique_special_chars'.format(feature)] = self.special_chars[feature]['count_unique']
                self.df['{}_clean_length'.format(feature)] = self.clean_phrases[feature]['count_len']
                self.df['{}_clean_count_words'.format(feature)] = self.clean_phrases[feature]['count_words']
                self.df['{}_clean_count_sents'.format(feature)] = self.clean_phrases[feature]['count_sents']
                self.df['{}_clean_count_conjuncts'.format(feature)] = self.clean_phrases[feature]['count_conjuncts']
                self.df['{}_pos_count_unique_tags'.format(feature)] = self.pos[feature]['count_unique_tags']
                self.df['{}_pos_count_unique_labels'.format(feature)] = self.pos[feature]['count_unique_labels']
                self.df['{}_pos_count_ad'.format(feature)] = self.pos[feature]['count_ad']
                self.df['{}_pos_count_verb'.format(feature)] = self.pos[feature]['count_verb']
                self.df['{}_pos_count_noun'.format(feature)] = self.pos[feature]['count_noun']
                self.df['{}_ner_recognized'.format(feature)] = self.ner[feature]['recognized']
                self.df['{}_ner_count'.format(feature)] = self.ner[feature]['count']
                self.df['{}_ner_count_unique'.format(feature)] = self.ner[feature]['count_unique']
                self.df['{}_ner_entity'.format(feature)] = self.ner[feature]['entity']
                self.df['{}_ner_entity_type'.format(feature)] = self.ner[feature]['entity_type']
                self.df['{}_dep_tree_count_has_child'.format(feature)] = self.dep[feature]['tree']['count_has_child']
                self.df['{}_dep_tree_count_has_child'.format(feature)] = self.dep[feature]['tree']['count_dep_of_child']
                self.df['{}_dep_tree_count_unique_dep_of_child'.format(feature)] = self.dep[feature]['tree']['count_unique_dep_of_child']
                self.df['{}_dep_tree_count_head_text_of_child'.format(feature)] = self.dep[feature]['tree']['count_head_text_of_child']
                self.df['{}_dep_tree_count_unique_head_text_of_child'.format(feature)] = self.dep[feature]['tree']['count_unique_head_text_of_child']
                self.df['{}_dep_tree_count_head_pos_of_child'.format(feature)] = self.dep[feature]['tree']['count_head_pos_of_child']
                self.df['{}_dep_tree_count_unique_head_pos_of_child'.format(feature)] = self.dep[feature]['tree']['count_unique_head_pos_of_child']
                self.df['{}_dep_tree_count_children'.format(feature)] = self.dep[feature]['tree']['count_children']
                #self.df['{}_dep_noun_pairs_count_children'.format(feature)] = self.dep[feature]['noun_pairs']['count_dep']
                self.df['{}_dep_noun_pairs_count_unique_dep'.format(feature)] = self.dep[feature]['noun_pairs']['count_unique_dep']
                self.df['{}_dep_noun_pairs_count_root_dep'.format(feature)] = self.dep[feature]['noun_pairs']['count_root_dep']
                self.df['{}_dep_noun_pairs_count_unique_root_dep'.format(feature)] = self.dep[feature]['noun_pairs']['count_unique_root_dep']
                self.df['{}_dep_noun_pairs_count_root_head_dep'.format(feature)] = self.dep[feature]['noun_pairs']['count_root_head_dep']
                self.df['{}_dep_noun_pairs_count_unique_root_head_dep'.format(feature)] = self.dep[feature]['noun_pairs']['count_unique_root_head_dep']
                self.df['{}_dep_noun_pairs_count_root_text_dep'.format(feature)] = self.dep[feature]['noun_pairs']['count_root_text_dep']
                self.df['{}_dep_noun_pairs_count_unique_root_text_dep'.format(feature)] = self.dep[feature]['noun_pairs']['count_unique_root_text_dep']
        self.emoji_handler(features=_features, has_emojis=True, count_emojis=True, count_unique_emojis=True, convert_emoji_to_text=True)

    def merge(self, features: List[str], sep: str = ' '):
        """
        Merge or concat text features

        :param features: List[str]
            Name of the features

        :param sep: str
            Special character to separate values
        """
        _features: List[str] = self.segments.get('enumeration') if features is None else features
        for first in _features:
            for second in _features:
                if first != second and '{}_merge_{}'.format(second, first) not in self.df.keys():
                    if first in self.df.keys():
                        if len(self.get_str_match(cases=list(self.df.keys()), substring='{}_merge_{}'.format(first, second))) == 0:
                            self.df['{}_merge_{}'.format(first, second)] = self.df[first].str.cat(others=self.df[second], sep=sep)
                            Log(write=False, level='info').log(msg='Merged "{}" and "{}" together (using {} as separator)'.format(first, second, sep))

    def replace(self, features: List[str], find_values: List[str], replace_value: str):
        """
        Replace text elements

        :param features: List[str]
            Name of the features

        :param find_values: List[str]
            Values to replace

        :param replace_value: str
            Value to replace with
        """
        if len(find_values) > 0:
            for feature in features:
                for val in find_values:
                    self.df[feature] = self.df[feature].str.replace(val, replace_value)

    def similarity(self, features: List[str] = None):
        """
        Calculate similarity of text

        :param features: List[str]
            Name of the features to calculate similarity of text feature
        """
        _features: List[str] = self.segments.get('phrases') if features is None else features
        self._apply_similarity(features=_features)
        for feature in self.similarity_scores.keys():
            _score_each_case: List[float] = []
            for case in self.similarity_scores[feature].keys():
                _score_each_case.append(sum(self.similarity_scores[feature][case]) / len(self.similarity_scores[feature][case]))
            self.df['{}_avg_similarity_score'.format(feature)] = _score_each_case

    def splitter(self, features: List[str] = None, sep: str = None):
        """
        Split text data by given separator

        :param features: List[str]
            Name of the features to split

        :param sep: str
            Separating character
        """
        _features: List[str] = self.segments.get('enumeration') if features is None else features
        for feature in _features:
            if feature in self.df.keys():
                if len(self.get_str_match(cases=list(self.df.keys()), substring='{}_split_'.format(feature))) == 0:
                    if sep is None:
                        _sep: str = self.enumeration.get(feature)
                    else:
                        _sep: str = sep
                    if _sep is None:
                        Log(write=False, level='info').log(msg='No separator found')
                    else:
                        _split_features: pd.DataFrame = self.df[feature].str.split(pat=sep, expand=True)
                        if _split_features.shape[1] <= 1:
                            Log(write=False, level='info').log(msg='Separator ({}) found in text feature "{}"'.format(_sep, feature))
                        else:
                            _split_features.rename(columns={i: '{}_split_{}'.format(feature, i) for i in range(0, _split_features.shape[1])})
                            self.df = pd.concat([self.df, _split_features], axis=1)
                            Log(write=False, level='info').log(msg='Feature "{}" split (by {}) into {} new features'.format(feature, sep, _split_features.shape[1]))

    def tfidf(self, features: List[str] = None):
        """
        Calculate term-frequency inverse data frequency

        :param features: List[str]
            Name of the features
        """
        _features: List[str] = self.segments.get('phrases') if features is None else features
        for feature in _features:
            self.cluster.update({feature: {}})
            self.df[feature] = self.df[feature].fillna('NaN')
            _tfidf_vectorizer: TfidfVectorizer = TfidfVectorizer(stop_words=LANG_MODELS.get(self.lang)['name'],
                                                                 ngram_range=(1, 3)
                                                                 )
            _clean_text_data: pd.DataFrame = self.df[feature].apply(lambda x: self._clean_text(phrase=str(x)))
            self.df['{}_tfidf'.format(feature)] = _tfidf_vectorizer.fit_transform(_clean_text_data.values).data[0:self.df.shape[0]]

    def translate(self, text: str, lang: str) -> str:
        """
        Translate given text (using Google Translate)

        :param text: str
            Text to translate

        :param lang: str
            Name of the target language

        :return: str
            Translated text
        """
        return self.translator.translate(text=text, dest=lang).text
