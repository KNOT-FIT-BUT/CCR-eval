import os

from ufal.morphodita import Tagger, Tokenizer
from ufal.morphodita import TaggedLemmas, TokenRanges, Forms 

class ModelLoadError(Exception):
    '''Custom exception for model loading error'''
    pass

class ModelNotLoadedError(Exception):
    '''Custom exception for not loaded model'''
    pass

class TokenizerError(Exception):
    '''Custom exception for tokenizer error'''
    pass

class Lemmatizer():

    MODEL_PATH = "models/czech-morfflex2.0-pdtc1.0-220710/czech-morfflex2.0-pdtc1.0-220710.tagger"
    MODEL_LOADED = False

    SENTENCE_ENDERS     =  [".", "!", "?"]
    RIGHT_DELIMITERS    = SENTENCE_ENDERS + [",", "“"]
    LEFT_DELIMITERS     = ["„"]

    LOAD_ERROR_STR = "Error while loading model"

    def __init__(self, model_path=MODEL_PATH):
        self.MODEL_PATH = model_path


    def load_model(self):
        if not os.path.exists(self.MODEL_PATH):
            raise FileNotFoundError
        
        self.tagger = Tagger.load(self.MODEL_PATH)
        
        try:
            self.morpho = self.tagger.getMorpho()
        except AttributeError:
            raise ModelLoadError(self.LOAD_ERROR_STR)

        if not self.tagger:
            raise ModelLoadError(self.LOAD_ERROR_STR)

        if not self.morpho:
            raise ModelLoadError(self.LOAD_ERROR_STR)
        
        self.MODEL_LOADED = True
          
    
    def tokenize_text(self, text:str) -> Tokenizer:
        '''Tokenizes the inputted text, returns instance of a Tokenzier()'''
        
        tokenizer = self.tagger.newTokenizer()

        if tokenizer is None:
            raise TokenizerError
        
        tokenizer.setText(text)
        return tokenizer


    def lemmatize_text(self, text:str) -> str:
        '''Performs text tokenization and returns the inputted text in a lemmatized format'''

        if not self.MODEL_LOADED:
            raise ModelNotLoadedError
        
        text_out = ""

        forms = Forms()
        lemmas = TaggedLemmas()
        tokens = TokenRanges()
        tokenizer = self.tokenize_text(text)
        
        sentence_end = False

        while tokenizer.nextSentence(forms, tokens):
            self.tagger.tag(forms, lemmas)

            for lemma in lemmas:
                token = self.morpho.rawLemma(lemma.lemma)
                if token in self.RIGHT_DELIMITERS:
                    text_out = text_out[:-1] + token +  " "
                    if token in self.SENTENCE_ENDERS:
                        sentence_end = True
                    continue

                if token in self.LEFT_DELIMITERS:
                    text_out += token
                    continue

                if sentence_end:
                    token = token.capitalize()
                    sentence_end = False

                text_out += token + " "

        return text_out 


