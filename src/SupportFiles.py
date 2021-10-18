# Copyright (c) 2017-present, 24/7 Customer Inc.
# All rights reserved.
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
import json
import os
import codecs

class SupportFiles:
    """This class parses config.json and produces corresponding supporting files"""
    DEFAULT_STOPWORDS_FILENAME = "stopwords.txt"
    DEFAULT_STEMWORDS_EXCEPTIONS_FILENAME = "stemmingException.txt"
    DEFAULT_CONTRACTIONS_FILENAME = "contractions.txt"

    STOP_WORDS_TRANSFORMATION = "stop-words"
    STEM_TRANSFORMATION = "stems"
    WORD_CLASS_SUBSTITUTION_REGEX = "wordclass-subst-regex"

    TRANSFORMATION_SECTION = "transformations"
    TRAINING_CONFIGS_SECTION = "trainingConfigs"
    STEMMING_EXCEPTION_FIELD_NAME = "stemmingExceptions"
    DEFAULT_INTENT_FIELD_NAME = "out_of_domain_intent"
    SPEECH_CONFIGS_SECTION = "speechConfigs"

    def __init__(self, inputConfigFile, workDirectory, logger):
        self.input_config_file = inputConfigFile
        self.work_directory = workDirectory
        self.stop_words_list = None
        self.contractions_pair_mappings = None
        self.stemming_exception_list = None
        self.word_classes_regex_list = {}
        self.default_intent = "None_None"
        self.training_configs = None
        self.speech_configs = None
        self.logger = logger

    def parse_config_file(self):
        """parsing config file and process all the transformations"""
        with codecs.open(self.input_config_file, "r", encoding='utf-8') as json_data:
            config_data = json.load(json_data)
            transform_configs = config_data[self.TRANSFORMATION_SECTION]
            for transform in transform_configs:
                if type(transform) is not str:
                    transform_name = list(transform)[0]
                    if transform[transform_name]["type"] in [self.STOP_WORDS_TRANSFORMATION]:
                        self.stop_words_list = transform[transform_name]["list"]   # Stopwords file
                    elif transform[transform_name]["type"] in [self.STEM_TRANSFORMATION]:
                        self.contractions_pair_mappings = transform[transform_name]["mappings"]  #Apostrophe words file
                    elif transform[transform_name]["type"] in [self.WORD_CLASS_SUBSTITUTION_REGEX]:
                        wordclass_pair_list = transform[transform_name]["mappings"]
                        self.word_classes_regex_list.update(wordclass_pair_list)
            self.training_configs = config_data[self.TRAINING_CONFIGS_SECTION]
            if self.SPEECH_CONFIGS_SECTION in config_data:
                self.speech_configs = config_data[self.SPEECH_CONFIGS_SECTION]
            if self.STEMMING_EXCEPTION_FIELD_NAME in self.training_configs:
                self.stemming_exception_list = self.training_configs[self.STEMMING_EXCEPTION_FIELD_NAME]# stemming exception File
            if self.DEFAULT_INTENT_FIELD_NAME in self.training_configs:
                self.default_intent = self.training_configs[self.DEFAULT_INTENT_FIELD_NAME]

        if self.stop_words_list is not None:
            with codecs.open(os.path.join(self.work_directory, self.DEFAULT_STOPWORDS_FILENAME), "w", encoding='utf-8') as file_handler:
                for item in self.stop_words_list:
                    file_handler.write("%s\n" % item)

        if self.contractions_pair_mappings is not None:
            with codecs.open(os.path.join(self.work_directory, self.DEFAULT_CONTRACTIONS_FILENAME), "w", encoding='utf-8') as file_handler:
                for key, value in self.contractions_pair_mappings.items():
                    file_handler.write(str(key) + '\t' + str(value) + '\n')

        if self.stemming_exception_list is not None:
            with codecs.open(os.path.join(self.work_directory, self.DEFAULT_STEMWORDS_EXCEPTIONS_FILENAME), "w", encoding='utf-8') as file_handler:
                for item in self.stemming_exception_list:
                    file_handler.write("%s\n" % item)

    def get_default_intent(self):
        """returns default intent to be used for model building"""
        return self.default_intent

    def get_stopwords_fileName(self):
        """returns stop words file path to be used for model building"""
        if self.stop_words_list is not None:
            return os.path.join(self.work_directory, self.DEFAULT_STOPWORDS_FILENAME)
        return None

    def get_stemming_exceptions_filename(self):
        """returns stemming exception words file path to be used for model building"""
        if self.stemming_exception_list is not None:
            return os.path.join(self.work_directory, self.DEFAULT_STEMWORDS_EXCEPTIONS_FILENAME)
        return None

    def get_contractions_file_name(self):
        """returns contractions words file path to be used for model building"""
        if self.contractions_pair_mappings is not None:
            return os.path.join(self.work_directory, self.DEFAULT_CONTRACTIONS_FILENAME)
        return None

    def get_training_configs_property(self, property_name):
        """returns all configurable training configs to be used for model building"""
        if self.training_configs is not None:
            if property_name in self.training_configs:
                return str(self.training_configs[property_name])
        return None

    def get_additional_training_configs_property(self):
        """returns configuration to be overidden over default configs"""
        self.logger.debug("Getting training configs")
        if self.training_configs is not None:
            additional_training_configs = dict(self.training_configs)
            del additional_training_configs['numOfEpochs']
            del additional_training_configs['validationSplit']
            del additional_training_configs['stemmingExceptions']
            return additional_training_configs
        return {}

    def get_speech_configs_section(self):
        """returns speech configuration"""
        if self.speech_configs is not None:
            return dict(self.speech_configs)
        return self.speech_configs