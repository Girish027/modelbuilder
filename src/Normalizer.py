# Copyright (c) 2017-present, 24/7 Customer Inc.
# All rights reserved.
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
import concurrent.futures
import logging
import os
import fnmatch
import codecs
import csv
import uuid
import ftfy
from unidecode import unidecode
from Web2nl import Web2nl
from config import data

class Normalizer:
    """This class submit job to web2nl executor for normalizing via web2nl and than write to a file to be
    submitted to DSG Java Workbench"""
    # No of worker threads to be used for sending requests to web2nl
    NO_OF_WORKER_THREADS = data.get("NO_OF_WORKER_THREADS",20)
    def __init__(self, normalizationUUID, modelUUID, logger, default_intent):
        self.web2nl_handler = Web2nl(normalizationUUID, logger)
        self.model_uuid = modelUUID
        self.logger = logger
        self.intent_mappings = {}
        self.normalize_text_mapping = {}
        logger.info("Normalization handler initialized successfully")
        self.default_intent = default_intent
        logger.info("Setting default intent to: %s", self.default_intent)


    def normalize_text(self):
        """Initialized Normalized output file format"""
        try:
            if self.web2nl_handler.is_model_valid() is False:
                raise RuntimeError("Failed in model validation, aborting Training")
            normalized_file_writer = logging.getLogger('log')
            normalized_file_writer.setLevel(logging.INFO)
            logpath = os.path.join(data['MODEL_BUILDER_STORAGE'], self.model_uuid) \
                      +"/FirstResponses_WCNormalized"
            file_handler = logging.FileHandler(logpath, "w", encoding = "utf-8")
            file_handler.setFormatter(logging.Formatter('%(message)s'))
            normalized_file_writer.addHandler(file_handler)

            speech_data_writer = logging.getLogger('speechLogger')
            speech_data_writer.setLevel(logging.INFO)
            logpath = os.path.join(os.path.join(data['MODEL_BUILDER_STORAGE'], self.model_uuid),
                                "corpus_1")
            file_handler = logging.FileHandler(logpath, "w", encoding = "utf-8")
            file_handler.setFormatter(logging.Formatter('%(message)s'))
            speech_data_writer.addHandler(file_handler)

            self.submit_job_to_web2nl(self.model_uuid, normalized_file_writer,speech_data_writer)
            return
        except Exception as exception:
            raise exception

    def write_to_file(self, normalized_text, intent, filename, source, normalized_file_writer):
        """Writing normalized data to FirstResponses_WCNormalized file"""
        if intent in self.intent_mappings:
            self.intent_mappings[intent] = self.intent_mappings[intent] + 1
        else:
            self.intent_mappings[intent] = 1
        normalized_text = ftfy.fix_text(normalized_text)
        normalized_text = normalized_text.replace(",", " ")
        '''
            Writing data in DSG workbench input format
            Filename |  Normalized Text | Level 1 intent | Level 2 intent
            Note: Level2 intent is only taken into account if hierarchical model is trained. For everything else it 
            is ignored
        '''
        normalized_file_writer.info("%s\t%s\t%s\t%s\t%s", filename, normalized_text, intent, intent, source)

    def fetch(self, transcription="test", intent="", filename="", source="I", normalized_file_writer=None, speech_data_writer=None):
        """Calling web2nl to get normalized text and than write to a file"""
        if transcription in self.normalize_text_mapping:
            normalized_text = self.normalize_text_mapping[transcription]
        else:
            normalized_text = self.web2nl_handler.get_normalize_text(transcription)
            self.normalize_text_mapping[transcription] = normalized_text
        self.write_to_file(normalized_text, intent, filename, source, normalized_file_writer)
        ## We dont want to write normalized text to speech file
        #speech_data_writer.info(normalized_text)
        return
    
    def normalizeSpeech(self, training_data_file):
        speech_data_writer = logging.getLogger('speechLogger')
        speech_data_writer.setLevel(logging.INFO)
        logpath = os.path.join(os.path.join(data.get('MODEL_BUILDER_STORAGE',None), self.model_uuid),
                                "corpus_1")
        self.logger.info("corpus_1 logPath : %s",str(logpath))
        self.logger.info("Normalize trainingFile passed to build Speech model")
        file_handler = logging.FileHandler(logpath, "w", encoding = "utf-8")
        file_handler.setFormatter(logging.Formatter('%(message)s'))
        speech_data_writer.addHandler(file_handler)
        utterance_count = 0
        self.logger.info("updating corpus_1 for speech model")
        try:
            with codecs.open(training_data_file, "r", encoding='utf-8') as csv_file :
                    read_csv = csv.DictReader(csv_file, delimiter='\t', fieldnames=("intent", "transcription",
                                                                                "Original Transcription",
                                                                                "Granular Intent", "Filename",
                                                                                "Transcription Hash", "Comments",
                                                                                "Source"))
                    is_header = False
                    for row in read_csv:
                        if not is_header:
                            is_header = True
                            continue
                        transcription = row['transcription']
                        if transcription:
                            speech_data_writer.info(transcription)
                        utterance_count = utterance_count+1
            self.logger.info("Total Utterances: %s", str(utterance_count))
        except Exception as exc:
            raise exc
        return

    def submit_job_to_web2nl(self, modeluuid, normalized_file_writer, speech_data_writer):
        """Submit each transcription to web2nl manager for sending it to web2nl"""
        future_to_normalized_text = None
        utterance_count = 0
        training_data_file = None
        try:
            for file in os.listdir(os.path.join(os.path.join(data['MODEL_BUILDER_STORAGE'], modeluuid), "data")):
                if fnmatch.fnmatch(file, 'input*'):
                    training_data_file = os.path.join(os.path.join(os.path.join(data['MODEL_BUILDER_STORAGE'],\
                                                                              modeluuid), "data"), file)
            self.logger.info("Starting to normalize data via web2nl. It may take some time depending on data size")
            with codecs.open(training_data_file, "r", encoding='utf-8') as csv_file, \
                concurrent.futures.ThreadPoolExecutor(max_workers=self.NO_OF_WORKER_THREADS) as executor:
                read_csv = csv.DictReader(csv_file, delimiter='\t', fieldnames=("intent", "transcription",
                                                                                "Original Transcription",
                                                                                "Granular Intent", "Filename",
                                                                                "Transcription Hash", "Comments",
                                                                                "Source"))
                is_header = False
                for row in read_csv:
                    if not is_header:
                        is_header = True
                        continue
                    intent = row['intent']
                    transcription = row['transcription']
                    if row.get('Source') is None:
                       source =  "I"
                    else:
                       source = row.get('Source')
                    if row.get('Filename') is None:
                       filename =  str(uuid.uuid4())
                    else:
                        filename = row.get('Filename')
                    if transcription:
                        speech_data_writer.info(transcription)
                        future_to_normalized_text = {
                            executor.submit(self.fetch, transcription, intent, filename, source,
                                            normalized_file_writer,speech_data_writer):transcription
                        }
                    utterance_count = utterance_count+1
            self.logger.info("Total Utterances: %s", str(utterance_count))
            if utterance_count is 0:
                self.logger.error("No transcription received for classification")
                raise RuntimeError("No transcription received for classification")
            for future in concurrent.futures.as_completed(future_to_normalized_text):
                try:
                    transcription = future.result()
                    self.logger.debug("Normalization completed for: %s", transcription)
                except Exception as exc:
                    self.logger.error("Something bad happened while normalizing data", exc)
                    raise exc
            if self.default_intent not in self.intent_mappings:
                default_transcription = "null"
                self.logger.warn("Since default intent was not found in training data, adding dummy transcription with "
                                 "intent: %s", self.default_intent)
                self.write_to_file(default_transcription, self.default_intent, "default-file", "I", normalized_file_writer)
            self.logger.info("Training a classifier model with intents: %s", self.intent_mappings)
            #self.logger.info("Training a classifier model with transcription mapping: %s", self.normalize_text_mapping)
        except Exception as exc:
            raise exc
