# Copyright (c) 2017-present, 24/7 Customer Inc.
# All rights reserved.
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
import os
import requests
from retrying import retry
from config import data

class Web2nl:
    """This class submit job to web2nl executor for normalizing via web2nl and than write to a file to be
        submitted to DSG Java Workbench"""
    def __init__(self, modelUUID, logger):
        self.logger = logger
        self.web2nl_url = data.get("WEB2NL_URL", None)
        self.web2nl_api_key = data.get("WEB2NL_API_KEY", None)
        if self.web2nl_url is None or self.web2nl_api_key is None:
            logger.error("Web2nl URL or API key not found")
            raise ValueError
        logger.info("Setting web2nl URL: %s, Web2nl API key: %s", self.web2nl_url, self.web2nl_api_key)
        self.model_builder_url = data.get("MODELBUILDER_URL", None)
        if self.model_builder_url is None:
            logger.error("Modelbuilder URL  not found")
            raise ValueError
        self.model_uuid = modelUUID

    @retry(wait_random_min=10, wait_random_max=600, stop_max_attempt_number=3)
    def is_model_valid(self):
        """Validates model with web2nl"""
        try:
            payload = {
                "modelurl": self.model_builder_url + self.model_uuid,
                "api_key": self.web2nl_api_key
            }

            response = requests.get(self.web2nl_url + "validations", params=payload)
            if response.status_code is requests.codes.no_content:
                return True
            elif response.status_code is requests.codes.bad_request:
                self.logger.error("Model validation failed: " + response.text)
                return False
            else:
                self.logger.error("Failed while validating model. Will retry in some time")
                raise RuntimeError("Failed while validating model")
        except requests.exceptions.ConnectionError as errc:
            self.logger.error("Error Connecting:", errc)

    @retry(wait_random_min=10, wait_random_max=600, stop_max_attempt_number=3)
    def get_normalize_text(self, input_string):
        """Normalizes data using web2nl"""
        try:
            payload = {
                "q": input_string,
                "modelurl":self.model_builder_url + self.model_uuid,
                "api_key": self.web2nl_api_key
            }
            response = requests.get(self.web2nl_url + "normalizations", params=payload)
            response.raise_for_status()
            response_in_json = response.json()
            text = response_in_json.get("lastInput", None)
            if text is None:
                raise RuntimeError("Failed getting normalized form from web2nl for: " + input_string)
            return text
        except Exception as err:
            raise err
