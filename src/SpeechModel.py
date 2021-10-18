# Copyright (c) 2019-present, 24/7 Customer Inc.
# All rights reserved.
import sh
import re
import os
import requests
from GenerateGRXML import GenerateGRXML
from config import data

class SpeechModel:
    """This class takes already created web2nl model and generate accoustic model based on Microsoft speech SDK"""

    def __init__(self, workDirectory, modelUUID, isUnbundled, digitalHostedUrl, logger):
        self.workDirectory = workDirectory
        self.modelId = modelUUID
        self.logger = logger
        self.isUnbundled = isUnbundled
        self.digitalHostedUrl = digitalHostedUrl
        self.grxml_handler = GenerateGRXML(workDirectory, modelUUID, logger)
        return

    def convertToArpa(self):
        """Converts input corpus to arpa format using srilm library"""
        try:
            self.logger.info("converting input corpus to arpa format")
            ngram_location = data.get("SRILM_TOOLS_LOCATION", None)
            if ngram_location is None or not os.path.exists(ngram_location):
                raise RuntimeError("SRILM tools are not available")
            cmd = sh.Command(ngram_location)
            cmd("-order", "3", "-text",
                self.workDirectory + "/Inter/normalizedCorpus.txt",
                "-kndiscount3", "-interpolate3", "-lm",
                self.workDirectory + "/Inter/normalizedCorpus.arpa")
        except Exception:
            raise RuntimeError("Failed while converting to arpa format. May be too little data")
        return

    def SubstituteClassesFromSubstitutionsFile(self):
        """substitutes word classes in input corpus from substitutions file"""
        self.logger.info("Substituting word classes from substitutions file")
        if os.path.isfile(self.workDirectory + "/word_classes.txt"):
            self.grxml_handler.readWordClasses(word_classes_file = self.workDirectory + "/word_classes.txt")
        else:
            self.grxml_handler.readWordClasses()
        input_corpus = self.workDirectory + "/corpus_1"
        norm_corpus = self.workDirectory + "/Inter/normalizedCorpus.txt"
        subs_file = self.workDirectory + "/grxml/substitutions_arpax"

        strnew = str.replace
        f = open(subs_file, 'r')
        patterns = f.read().splitlines()
        f.close()

        fo = open(norm_corpus, 'w')
        fp = open(input_corpus, 'r')
        raw_string = r'\b(?:' + '|'.join(sorted(self.grxml_handler.word_classes_new.keys(), reverse=True)) + r")'?s?\b"
        robj = re.compile(raw_string)

        for line in fp.readlines():
            line = line.strip()
            line = robj.sub(
                lambda m: self.grxml_handler.word_classes_new[m.group(0)] if
                m.group(0) in self.grxml_handler.word_classes_new
                else self.grxml_handler.word_classes_new[
                    m.group(0)[:-1].replace("'", "")], line)
            wds = line.split()
            for p in patterns:
                temp = p.split()
                find = r'\b' + temp[0] + r'\b'
                replace = temp[1]
                line = re.sub(find, replace, line)

            if (len(wds) >= 2):
                if (wds[0] == "<s>" and wds[len(wds) - 1] == "</s>"):
                    fo.write(line + "\n")
                    continue
            fo.write(line + "\n")

        fp.close()
        fo.close()
        return


    def addBackoff(self):
        """This is adding back-off weights to arpa file"""
        self.logger.info("Adding backoff weights to normalized arpa corpus")
        input_arpa_file = self.workDirectory + "/Inter/normalizedCorpus.arpa"
        fp = open(input_arpa_file, 'r')
        op1 = open(self.workDirectory + '/Inter/inter.arpa', 'w')

        flag = 0
        gram_flag = 0
        max_ngram = 3
        for i in fp.readlines():
            newline = i.strip()
            fstr = newline.split()
            if (flag > 0):
                if (fstr[1] == "</s>"):
                    val = float(fstr[0]) * 2
                    val = "%.6f" % val
                    op1.write(str(val) + ' ' + fstr[1] + ' 0' + '\n')
                    flag = flag - 1
                    continue
                elif (fstr[1] == "<s>"):
                    op1.write(str(val) + ' ' + fstr[1] + ' 0' + '\n')
                    flag = flag - 1
                    continue
            if newline.find("1-grams:") == 1:
                flag = 2
            if newline.find("1-grams:") == 1:
                gram_flag = 1
            if newline.find("2-grams:") == 1:
                gram_flag = 2
            if newline.find("3-grams:") == 1:
                gram_flag = 3

            gram_dim = gram_flag + 2
            if (gram_flag < max_ngram and len(fstr) != gram_dim and len(fstr) > 1):
                op1.write(newline + " 0" + "\n")
            else:
                op1.write(newline + '\n')

        fp.close()
        op1.close()

    def compileSLMModel(self):
        """This function will compile SLM model and combine it with existing SSI Model"""
        userName = "mbs"
        if 'MICROSOFT_SDK_HOST' not in data or 'RECO_HOST' not in data:
            raise RuntimeError("Microsoft SDK HOST or Speech recognition host not configured!!")
        windowsSpeechServerHost = data.get("MICROSOFT_SDK_HOST")
        windowsSpeechServer = sh.ssh.bake("-l",userName,windowsSpeechServerHost)
        sshURL = userName + "@" + windowsSpeechServerHost
        tmp_dir = "\\Users\\" + userName + "\\" + self.modelId

        ''' Create tmp directory and copy required files to that directory '''
        command = "New-Item -type directory " + tmp_dir + " | Out-Null;"
        self.logger.info("Creating new temp directory and copying arpa, recoconfig and substitutions file")
        windowsSpeechServer(command)


        # In case of unbundled we create accoustic model with external.arpax However, in case of bundled,
        # we use all.arpax
        if self.isUnbundled == "true":
            sh.scp(self.workDirectory + "/Inter/inter.arpa",
               "./supporting_files/RecoConfig.xml",
               self.workDirectory + "/grxml/external.arpax",
               sshURL + ":" + tmp_dir)

            self.logger.info("Creating Accoustic model using Microsoft SDK")
            command = "Push-Location C:\\Users\\" + userName + "\\" + self.modelId + ";&(\'C:\\Program Files\\Microsoft " \
                                                                           "SDKs\\Speech\\v11.1\\Tools\\CompileGrammar.exe\') -In \\Users\\" + userName +"\\" + self.modelId + "\\external.arpax " \
                                                                                                                                                                          "-InFormat ARPA -Out inter.cfg " \
                                                                                                                                                                  "-RecoConfig RecoConfig.xml"
        else:
            sh.scp(self.workDirectory + "/Inter/inter.arpa",
                   "./supporting_files/RecoConfig.xml",
                   self.workDirectory + "/grxml/all.arpax",
                   sshURL + ":" + tmp_dir)

            self.logger.info("Creating Accoustic model using Microsoft SDK")
            command = "Push-Location C:\\Users\\" + userName + "\\" + self.modelId + ";&(\'C:\\Program " \
                                                                                     "Files\\Microsoft " \
                                                                                     "SDKs\\Speech\\v11.1\\Tools\\CompileGrammar.exe\') -In \\Users\\" + userName + "\\" + self.modelId + "\\all.arpax " \
                                                                                                                                                                                          "-InFormat ARPA -Out inter.cfg " \
                                                                                                                                                                                          "-RecoConfig RecoConfig.xml"
        windowsSpeechServer(command)

        ''' Copying Accoustic model to modelbuilder service '''
        sh.scp(sshURL + ":" + "\\Users\\" + userName + "\\" + self.modelId + "\\inter.cfg",
               self.workDirectory + "/Inter/inter.cfg")
        self.logger.info("Accoustic model created successfully")

        os.system("chmod 644 " + self.workDirectory + "/Inter/inter.cfg")

        """ Deleting the temp directory """
        self.logger.debug("Deleting the temp directory on Windows machine")
        command = "Remove-Item -Force -Recurse " + tmp_dir
        windowsSpeechServer(command)

    def combineSLMModelWithSSI(self):
        ''' This is to combine SLM and SSI Model'''
        userName = "mbs"
        self.logger.info("Combining SLM and SSI model into compiled SLM+SSI model on reco machine")

        recoHost = data.get("RECO_HOST")
        recoServer = sh.ssh.bake("-l", userName, recoHost)
        recoSSHUrl = userName + "@" + recoHost
        tmp_dir = "\\Users\\" + userName + "\\" + self.modelId

        command = "New-Item -type directory " + tmp_dir + " | Out-Null;Copy-Item " \
                                                          "C:\\tools\\local\\tellme\\bin\\Microsoft.Tellme.classifiermodelcompilerAssembly\\libtmexpat.dll " + tmp_dir + " | Out-Null;"
        recoServer(command)

        sh.scp(self.workDirectory + "/Inter/inter.cfg",
               self.workDirectory + "/final_model/web2nl/ssi_1",
               "./supporting_files/classifier_params",
               recoSSHUrl + ":" + tmp_dir)

        command = "Push-Location " + tmp_dir + ";C:\\tools\\local\\tellme\\bin\\classifiermodelcompiler.exe -m ssi_1 " \
                                               "-p " \
                                               "classifier_params -o " + tmp_dir + "\\ssi.cfr;"
        recoServer(command)

        command = "Push-Location " + tmp_dir + ";C:\\tools\\local\\tellme\\bin\\cfgappend.exe inter.cfg ssi.cfr " \
                                               "compiled_model.cfg;"
        recoServer(command, _ok_code=[0, 1])

        sh.scp(recoSSHUrl + ":" + tmp_dir +
               "\\compiled_model.cfg",
               self.workDirectory + "/final_model/speech/compiled_model.cfg")

        os.system("chmod 644 " + self.workDirectory + "/final_model/speech/compiled_model.cfg")

        ''' Cleaning up the temp directory '''
        self.logger.info("Deleting the temp directory on reco machine")
        command = "Remove-Item -Force -Recurse " + tmp_dir
        recoServer(command)

    def createWeb2nlFile(self):
        """This will generate web2nl files with URL pointed to actual modelbuilder service"""
        # TODO: we can substitute here actual deployed model link if available (this is for production)
        self.logger.info("Using Modelbuilder service hosted digital model")
        with open('./supporting_files/ssi_1', mode='r') as f:
            contents = f.read()

        if 'MODELBUILDER_URL' not in data:
            raise RuntimeError("Modelbuilder URL is not configured")

        final_model = ''
        if(not (self.digitalHostedUrl and self.digitalHostedUrl.strip())):
            contents += data.get('MODELBUILDER_URL') + self.modelId + "/digital"
        else:
            contents += self.digitalHostedUrl
            final_model = requests.get(self.digitalHostedUrl).text

        if not os.path.exists(self.workDirectory + "/final_model/web2nl"):
            os.system("mkdir -p " + self.workDirectory + "/final_model/web2nl")
        #contents+= "http://anvil.tellme.com/~kcortes/capone_bank/20190926/root_intent.model"
        with open(self.workDirectory + '/final_model/web2nl/ssi_1', mode='w') as f:
            f.write(contents)
        if(final_model != ''):
            with open(self.workDirectory + '/final_model/web2nl/final.model', mode='w') as f:
                f.write(final_model)

        return

    def zipIfUnbundled(self):
        if self.isUnbundled == "false":
            os.system("rm -rf " + self.workDirectory + "/final_model/model")
            sh.mkdir("-p", self.workDirectory + "/final_model/model" )
            sh.mkdir("-p", self.workDirectory + "/final_model/model/grxml")
            for filename in os.listdir(self.workDirectory + "/grxml"):
                if filename.endswith(".grxml"):
                    sh.cp("-r", self.workDirectory + "/grxml/" + filename ,self.workDirectory + "/final_model/model/grxml/.")
            sh.cp(self.workDirectory + "/final_model/speech/compiled_model.cfg",self.workDirectory +
                  "/final_model/model/.")
            basePath = os.getcwd()
            sh.cd(self.workDirectory + "/final_model")
            sh.zip("-r", os.path.join(self.workDirectory, "final_model/model.zip"), os.path.join(
                "./model"))
            sh.cd(basePath)

    def zipSLMIfUnbundled(self):
        if self.isUnbundled == "false":
            os.system("rm -rf " + self.workDirectory + "/slm_model/model")
            sh.mkdir("-p", self.workDirectory + "/slm_model/model" )
            sh.mkdir("-p", self.workDirectory + "/slm_model/model/grxml")
            for filename in os.listdir(self.workDirectory + "/grxml"):
                if filename.endswith(".grxml"):
                    sh.cp("-r", self.workDirectory + "/grxml/" + filename ,self.workDirectory + "/slm_model/model/grxml/.")
            sh.cp(self.workDirectory + "/Inter/inter.cfg",self.workDirectory +
                  "/slm_model/model/.")
            basePath = os.getcwd()
            sh.cd(self.workDirectory + "/slm_model")
            sh.zip("-r", os.path.join(self.workDirectory, "slm_model/model.zip"), os.path.join(
                "./model"))
            sh.cd(basePath)

    def generateCombinedModel(self):
        """Method to invoked for generating combined model"""
        try:
            self.logger.info("Starting to generate combined model with uuid: " + self.modelId)
            self.createWeb2nlFile()
            self.combineSLMModelWithSSI()
            self.zipIfUnbundled()
        except Exception:
            raise

    def generateSLMModel(self):
        """Method invoked for generating SLM model"""
        try:
            self.logger.info("Starting to generate SLM model with uuid: " + self.modelId)
            sh.mkdir("-p", self.workDirectory + "/Inter")
            sh.mkdir("-p", self.workDirectory + "/final_model/speech")
            self.SubstituteClassesFromSubstitutionsFile()
            self.convertToArpa()
            self.addBackoff()
            self.compileSLMModel()
            self.zipSLMIfUnbundled()
        except Exception:
            raise

    def generateSpeechModel(self):
        """Main method invoked for generating speech model"""
        try:
            self.logger.info("Starting to generate speech model with uuid: " + self.modelId)
            self.generateSLMModel()
            self.generateCombinedModel()
        except Exception:
            raise