import numpy as np
from numpy import random
import spacy
from spacy.attrs import LOWER, POS, ENT_TYPE, IS_ALPHA
from spacy.tokens import Doc
from spacy.matcher import Matcher
import contextualSpellCheck
import pandas as pd
import joblib

# nlp = spacy.load('en_core_web_sm')

# caller_name = None
# app_name = None
# phone_no = None
# time = None

class Helper:

    nlp = spacy.load('en_core_web_sm', disable=["ner","textcat"])
    matcher = Matcher(nlp.vocab)

    caller_name = None
    app_name = None
    phone_no = None
    time = None
    chat_count = 0

    def __init__(self):
        # Add Name Matchers
        pattern1 = [{'POS': "AUX"},
                   {'POS': "PROPN"}]
        pattern2 = [{'POS': "AUX"},
                    {'DEP': "acomp"}]
        pattern3 = [{'POS': 'AUX'},
                    {'DEP': 'attr'}]
        pattern4 = [{'POS': "PROPN"},
                    {'LOWER': 'here'}]

        self.matcher.add("Caller_Name1", None, pattern1)
        self.matcher.add("Caller_Name2", None, pattern2)
        self.matcher.add("Caller_Name3", None, pattern3)
        self.matcher.add("Caller_Name4", None, pattern4)

    def preprocess(self,text):
        doc = self.nlp(text)
        # [t.lemma_ for t in doc if not t.is_stop if t.lemma_.isalpha()]
        # [t.text.title() if t.pos_ == 'ADJ' else t.text for t in doc]
        lemma_list = [t.lemma_ for t in doc]
        return self.listToString(lemma_list)
    
    def remove_names(self,text):
        doc = self.nlp(text.lower())
        self.matches = self.matcher(doc)
        
        for match_id, start, end in self.matches:   
            string_id = self.nlp.vocab.strings[match_id]
            if string_id == "Caller_Name1" or string_id == "Caller_Name2" or string_id == "Caller_Name3":
                return self.remove_span(doc, end-1)
            elif string_id == "Caller_Name4":
                return self.remove_span(doc,start)
            else:
                return text
    
    def listToString(self,s):  
        str1 = " " 
        return (str1.join(s))
    
    def remove_span(self, doc, index):
        np_array = doc.to_array([LOWER, POS, ENT_TYPE, IS_ALPHA])
        np_array_2 = np.delete(np_array, (index), axis = 0)
        doc2 = Doc(doc.vocab, words=[t.text for i, t in enumerate(doc) if i!=index])
        doc2.from_array([LOWER, POS, ENT_TYPE, IS_ALPHA], np_array_2)
        return str(doc2)
    

    def check_name(self,text):
        doc = self.nlp(text.lower())
        self.matches = self.matcher(doc)
        
        for match_id, start, end in self.matches:   
            string_id = self.nlp.vocab.strings[match_id]
            if string_id == "Caller_Name1" or string_id == "Caller_Name2" or string_id == "Caller_Name3":
                self.caller_name = str(doc[end-1]).title()
            elif string_id == "Caller_Name4":
                self.caller_name = str(doc[start]).title()
            print('Chat Person name is :- ', self.caller_name)


    ########## Chat Helper ########
    def get_general_greeting(self):
        greet_data = pd.read_excel('GeneralGreetinf.xlsx', header=0) 
        item = random.randint(len(greet_data))
        if self.caller_name != None:
            return greet_data['Chat'][item].replace("USER", self.caller_name)
        else:
            return greet_data['Chat'][item].replace("USER", "")
    
    def get_chat_response(self,text):
        # load the model from disk
        loaded_model = joblib.load("Group_model.sav")
        self.check_name(text)
        pred_group = loaded_model.predict([self.preprocess(text)])
        print("Pred Group is ", pred_group)
        if pred_group == 1:
            return self.get_general_greeting()
