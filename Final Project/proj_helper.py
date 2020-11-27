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

    welcome_greeting = None
    

    def __init__(self):
        # Add Name Matchers
        name_pattern_1 = [{'POS': 'AUX', 'OP': '+'},
                            {'POS': 'PROPN', 'OP': '*'},
                            {'POS': 'PROPN', 'OP': '!', 'DEP': 'compound'}]
        name_pattern_2 = [{'POS': "PROPN"},
                            {'LOWER': 'here'}]
        
        # Add General Greeting Matchers
        greeting_pattern_1 = [{'LOWER': 'good', 'OP': '*'},
                                {'LOWER': 'morning'}]
        greeting_pattern_2 = [{'LOWER': 'good', 'OP': '*'},
                                {'LOWER': 'evening'}]
        greeting_pattern_3 = [{'LOWER': 'good', 'OP': '*'},
                                {'LOWER': 'afternoon'}]
        
        

        self.matcher.add("Caller_Name1", None, name_pattern_1)
        self.matcher.add("Caller_Name2", None, name_pattern_2)
        self.matcher.add("Greeting1", None, greeting_pattern_1)
        self.matcher.add("Greeting2", None, greeting_pattern_2)
        self.matcher.add("Greeting3", None, greeting_pattern_3)

    def preprocess(self,text):
        doc = self.nlp(text)
        # [t.lemma_ for t in doc if not t.is_stop if t.lemma_.isalpha()]
        # [t.text.title() if t.pos_ == 'ADJ' else t.text for t in doc]
        lemma_list = [t.lemma_ for t in doc if t.pos_ != 'PUNCT']
        return self.listToString(lemma_list)
    
    def remove_names(self,text):
        if self.caller_name != None:
            new_set = text.lower().replace(self.caller_name.lower(), '-username-')
            return new_set
        else:
            return text
    
    def remove_greetings(self,text):
        if self.welcome_greeting != None:
            new_set = text.lower().replace(self.welcome_greeting.lower(), '-greeting-')
            return new_set
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
        doc = self.nlp(text)
        self.matches = self.matcher(doc)
        
        for match_id, start, end in self.matches:   
            string_id = self.nlp.vocab.strings[match_id]
            if string_id == "Caller_Name1":
                self.caller_name = str(doc[start+1:end]).title()
                break
            elif string_id == "Caller_Name2":
                self.caller_name = str(doc[start]).title()
                break
        print('Chat Person name is :- ', self.caller_name)

    def remove_unwanted_text(self,text):
        removed_name = self.remove_names(text)
        removed_greeting = self.remove_greetings(removed_name)
        return removed_greeting

    def check_greeting(self,text):
        doc = self.nlp(text)
        self.matches = self.matcher(doc)
        
        for match_id, start, end  in self.matches:   
            string_id = self.nlp.vocab.strings[match_id]
            if string_id == "Greeting1":
                self.is_welcome_greeting = True
                self.welcome_greeting = "Good Morning"
                print('Detected Time Greeting!')
                break
            elif string_id == "Greeting2":
                self.is_welcome_greeting = True
                self.welcome_greeting = "Good Evening"
                print('Detected Time Greeting!')
                break
            elif string_id == "Greeting3":
                self.is_welcome_greeting = True
                self.welcome_greeting = "Good Afternoon"
                print('Detected Time Greeting!')
                break


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
        self.check_greeting(text)
        proc_text = self.remove_unwanted_text(text)
        print('Test ||| ', self.preprocess(proc_text))
        pred_group = loaded_model.predict([self.preprocess(proc_text)])
        print("Pred Group is ", pred_group)
        if pred_group == 1:
            return self.get_general_greeting()
