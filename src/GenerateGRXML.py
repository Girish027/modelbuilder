# Copyright (c) 2017-present, 24/7 Customer Inc.
# All rights reserved.
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
import os
import codecs

class GenerateGRXML:
    """This class generates GRXML files from static word classes file"""
    def __init__(self, workDirectory, modelUUID, logger):
        self.word_classes = {}
        self.word_classes_new = {}
        self.modelUUID = modelUUID
        self.workDirectory = workDirectory
        self.logger = logger
        return

    def generateRootGRXML(self, lang='en-US'):
        """Generate Master Index for all GRXML files"""
        grxml_file = open(os.path.join(self.workDirectory, 'word_classes_root.grxml'), 'w')

        ## Write grxml meta information
        grxml_file.write("<?xml version= \"1.0\"?>\n")
        grxml_file.write(
            "<grammar tag-format=\"semantics/1.0\" version=\"1.0\" xml:lang=\"" + lang + "\" "
                                                                                         "xmlns=\"http://www.w3.org/2001/06/grammar\">\n")

        for word_class in self.word_classes.keys():
            grxml_file.write("\t<rule id=\"" + word_class + "\" scope=\"public\">\n")
            grxml_file.write("\t\t<item>\n")
            grxml_file.write(
                "\t\t\t<ruleref uri=\"" + "." + "/" + word_class.replace("_class_",
                                                                         "") +
                ".grxml\"/>\n")
            grxml_file.write("\t\t</item>\n")
            grxml_file.write(
                "\t\t<tag>gSubstitutions.push({interpretation: rules.latest(),tokens: meta.latest().text,klass: \"" +
                word_class + "\" });\n")
            grxml_file.write("\t\t</tag>\n")
            grxml_file.write("\t</rule>\n")

        grxml_file.write("</grammar>")

        grxml_file.close()

    def generateClassGrammar(self, class_name, lang='en-US'):
        """Generate grxml files for word classes"""
        ## Check whether class contains any class elements
        if len(self.word_classes.get(class_name)) == 0:
            raise Exception("Class " + class_name + " doesn't contain any class elements!!")

        grxml_file = codecs.open(
            os.path.join(self.workDirectory, class_name.replace("_class_", "") + '.grxml'), 'wb', encoding='utf-8')

        ## Write grxml meta information
        grxml_file.write("<?xml version= \"1.0\"?>\n")
        grxml_file.write("<grammar mode=\"voice\" root=\"" + class_name.replace("_class_",
                                                                                "") + "\" "
                                                                                      "tag-format=\"semantics/1.0\" "
                                                                                      "version=\"1.0\" xml:lang=\"" +
                         lang + "\" xmlns=\"http://www.w3.org/2001/06/grammar\">\n")
        grxml_file.write("<!-- Machine generated - DO NOT EDIT -->\n")
        grxml_file.write("<!-- " + class_name + " -->\n")
        grxml_file.write("\t<rule id=\"" + class_name.replace("_class_", "") + "\" scope=\"public\">\n")
        grxml_file.write("\t  <item>\n")
        grxml_file.write("\t    <one-of>\n")

        ## Write class elements
        for class_element in self.word_classes.get(class_name):
            grxml_file.write("\t      <item>\n")
            grxml_file.write("\t        " + class_element + "\n")
            grxml_file.write("\t      </item>\n")

        grxml_file.write("\t    </one-of>\n")
        grxml_file.write("\t  </item>\n")
        grxml_file.write("\t</rule>\n")
        grxml_file.write("</grammar>")
        grxml_file.close()

    def readWordClasses(self, word_classes_file="supporting_files/word_classes.txt"):
        """This will read all the word classes from word_classes.txt file"""
        word_class_file = codecs.open(word_classes_file, 'rb', encoding='utf-8')
        # this map is hold a word as the key & a list of word classes it is associated with as the value,
        # this is just to do a validation. A word should be mapped to only one word class
        word_and_classes = {}

        for line in word_class_file:
            line = line.strip()
            ## Discard blank lines
            if line == "":
                continue
            ## Read class names
            elif line.startswith('_class_'):
                current_word_class = line
                self.word_classes[current_word_class] = set([])
                continue

            list_of_word_classes = word_and_classes[line] if line in word_and_classes else []
            list_of_word_classes.append(current_word_class)
            word_and_classes[line] = list_of_word_classes

            self.word_classes[current_word_class].add(line)
            self.word_classes_new[line] = current_word_class

        multiple_entries = {word: classes for word, classes in word_and_classes.items() if len(classes) > 1}
        if (len(multiple_entries) > 0):
            self.logger.warn('mulitple entries are present for some words in the word class file , correct them & '
                             're-run the script.')
        for word, classes in multiple_entries.items():
            self.logger.warn('Multiple entries present for the word ' + word + ' .They are in classes :- ' +
                             ','''.join(classes))
        self.logger.info("Read " + str(len(self.word_classes)) + " word classes")


    def createSubFile(self):
        """Generate substitutions files to be used with Microsoft SDK v11.1"""
        input_file = "supporting_files/word_classes.txt"
        output_file = self.workDirectory + "/substitutions.txt"
        sentence = "<URL:word_class_grammars/word_classes_root.grxml"
        fin = open(input_file, "r")
        fout = open(output_file, "w")
        for line in fin:
            line = line.strip()
            if (line.startswith("_class_")):
                toWrite = line + " " + sentence + "#" + line + ">\n"
                fout.write(toWrite)

    def generate_arpax(self, lang='en-US'):
        """Generate arpax files to be used with Microsoft SDK v11.1"""
        arpaxFile = codecs.open(os.path.join(self.workDirectory, 'all.arpax'), 'w', encoding='utf-8')
        substitutions_file = codecs.open(os.path.join(self.workDirectory, 'substitutions_arpax'), 'w',
                                         encoding='utf-8')

        arpaxFile.write(
            "<language-model xml:lang=\"" + lang + "\" root=\"TopLevelRule\" tag-format=\"semantics/1.0\">\n\n")
        arpaxFile.write("<ngram-rule scope=\"public\" id=\"BasicNGram\" src=\"inter.arpa\">\n")

        for word_class in self.word_classes.keys():
            arpaxFile.write(
                "\t<ngram-token id=\"" + word_class + "\" type=\"ruleref\" value=\"#" + word_class + "_1\"/>\n")
            substitutions_file.write(word_class + " <#" + word_class + ">\n")

        arpaxFile.write("</ngram-rule>\n\n")

        arpaxFile.write("<rule id=\"TopLevelRule\" scope=\"public\">\n")
        arpaxFile.write(
            "    <tag> out={}; index=0; out.Fragments=FragsGlobal = new Array(); out.classsubstitution = "
            "gSubstitutions = [];</tag>\n")
        arpaxFile.write("    <ruleref uri=\"#BasicNGram\"/>\n")
        arpaxFile.write("</rule>\n\n")
        for word_class in self.word_classes.keys():
            arpaxFile.write("<rule id=\"" + word_class + "_1\" scope=\"public\">\n")
            arpaxFile.write("\t<ruleref uri=\"#" + word_class + "_2\"/>\n")
            arpaxFile.write(
                "\t<tag> gSubstitutions.push({interpretation: rules.latest(),tokens: meta.latest().text,klass: \"" +
                word_class + "\" }); </tag>\n")
            arpaxFile.write("</rule>\n\n")

        ## grxmls for individual classes
        for word_class in self.word_classes.keys():
            arpaxFile.write("<rule id=\"" + word_class + "_2\" scope=\"public\">\n")
            arpaxFile.write("\t<one-of>\n")

            for element in self.word_classes.get(word_class):
                arpaxFile.write("\t    <item>\n")
                arpaxFile.write("\t    " + element + "\n")
                arpaxFile.write("\t    </item>\n")

            arpaxFile.write("\t</one-of>\n")
            arpaxFile.write("</rule>\n\n")
        arpaxFile.write("</language-model>")

    def generate_arpax2(self, lang='en-US'):

        arpaxFile = codecs.open(os.path.join(self.workDirectory, 'external.arpax'), 'w', encoding='utf-8')
        substitutions_file = codecs.open(os.path.join(self.workDirectory, 'substitutions_arpax'), 'w',
                                         encoding='utf-8')
        arpaxFile.write(
            "<language-model xml:lang=\"" + lang + "\" root=\"TopLevelRule\" tag-format=\"semantics/1.0\">\n\n")
        arpaxFile.write("<ngram-rule scope=\"public\" id=\"BasicNGram\" src=\"inter.arpa\">\n")

        for word_class in self.word_classes.keys():
            arpaxFile.write(
                "\t<ngram-token id=\"" + word_class + "\" type=\"ruleref\" value=\"#" + word_class + "_1\"/>\n")
            substitutions_file.write(word_class + " <#" + word_class + ">\n")

        arpaxFile.write("</ngram-rule>\n\n")

        arpaxFile.write("<rule id=\"TopLevelRule\" scope=\"public\">\n")
        arpaxFile.write(
            "    <tag> out={}; index=0; out.Fragments=FragsGlobal = new Array(); out.classsubstitution = "
            "gSubstitutions = [];</tag>\n")
        arpaxFile.write("    <ruleref uri=\"#BasicNGram\"/>\n")
        arpaxFile.write("</rule>\n\n")
        for word_class in self.word_classes.keys():
            arpaxFile.write("<rule id=\"" + word_class + "_1\" scope=\"public\">\n")
            arpaxFile.write(
                "\t<ruleref uri=\"" + "./grxml/" + word_class.replace("_class_",
                                                                                                       "") + ".grxml\"/>\n")
            arpaxFile.write(
                "\t<tag> gSubstitutions.push({interpretation: rules.latest(),tokens: meta.latest().text,klass: \"" + word_class + "\" }); </tag>\n")
            arpaxFile.write("</rule>\n\n")

        arpaxFile.write("</language-model>")

    def generateGrammerFiles(self):
        """Main method to generate all the grxml and arpax files"""
        self.logger.info("Reading default word classes file")
        if os.path.isfile(self.workDirectory + "/word_classes.txt"):
            self.readWordClasses(word_classes_file = self.workDirectory + "/word_classes.txt")
        else:
            self.readWordClasses()
        self.workDirectory = self.workDirectory + "/grxml"
        for word_class in self.word_classes.keys():
            self.logger.info("Generating grammer file for " + word_class)
            self.generateClassGrammar(word_class, lang="en-US")
        self.logger.info("Generating Root Grammar file")
        self.generateRootGRXML(lang="en-US")

        # For SDK 11.1
        self.logger.info("Generating substitution file")
        self.createSubFile()
        self.generate_arpax()
        self.generate_arpax2()
        return



