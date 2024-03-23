import torch
from torchvision.transforms import v2
import gensim
import nltk
import re
import contractions
nltk.download('wordnet')


class TextPreprocessing():
    def __init__(self ,
                 regexList = None,
                 punct= True,
                 lowercase= True,
                 slang= False,
                 stopwordList = None,
                 stemming = False,
                 lemmatization= False ):

        self.convertToLowercase = lowercase #Done
        self.removePunctuations = punct #Done
        self.regexList = regexList  # Done
        self.removeSlang = slang #Done
        self.stopwordList = stopwordList #Done
        self.useStemming = stemming #Done
        self.useLemmatization = lemmatization #Done

    def process(self , text):
        # Make text lower case
        if self.convertToLowercase:
            text = text.lower()

        pattern = r"\s*\([a-zA-Z]\s_\)"
        text = re.sub(pattern, "", text)

        #Convert multiline with spaces
        text = text.replace("\n", " ")

        if self.removeSlang:
            text = contractions.fix(text)

        #Remove punctuations
        if self.removePunctuations:
            text = re.sub(r"[=.!,¿?.!+,;¿/:|%()<>।॰{}#_'\"@$^&*']", " ", text)
            text = re.sub(r"…", " ", text)

        # remove double quotes
        text = re.sub(r'"', " ", text)

        # remove numbers
        text = re.sub(r'[0-9]', "", text)
        # sentence = re.sub(r'#([^s]+)', r'1', sentence)

        # remove website links
        text = re.sub('((www.[^s]+)|(https?://[^s]+))', '', text)

        # remove multiple spaces
        text = re.sub(r'[" "]+', " ", text)

        # remove extra space
        text = text.strip()

        if self.regexList is not None:
            for regex in self.regexList:
                text = re.sub(regex, '', text)

        if self.stopwordList is not None:
            text_list = text.split()
            text_list = [word for word in text_list if word not in self.stopwordList]
            text = " ".join(text_list)

        #Stemming (convert the word into root word)
        if self.useStemming:
            ps = nltk.stem.porter.PorterStemmer()
            text_list = text.split()
            text_list = [ps.stem(word) for word in text_list]
            text = " ".join(text_list)

        #Lemmatization (convert the word into root word)
        if self.useLemmatization:
            lem = nltk.stem.wordnet.WordNetLemmatizer()
            text_list = text.split()
            text_list = [lem.lemmatize(word) for word in text_list]
            text = " ".join(text_list)

        return text

# preprocessor = TextPreprocessing(slang=True)

# sample = "I can't do this"

# print(f"Before preprocessing :{sample}")
# print(f"After preprocessing :{preprocessor.process(sample)}")

