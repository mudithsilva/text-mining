import numpy as np
from numpy import random
import spacy
from spacy.vocab import Vocab
from spacy_lookup import Entity
from spacy.attrs import LOWER, POS, ENT_TYPE, IS_ALPHA
from spacy.tokens import Doc
from spacy.matcher import Matcher
import contextualSpellCheck
import pandas as pd
import joblib

import datetime

# nlp = spacy.load('en_core_web_sm')

# caller_name = None
# app_name = None
# phone_no = None
# time = None

class Helper:

    nlp = spacy.load('en_core_web_sm', disable=["ner","textcat"])
    # load the model from disk
    loaded_model = joblib.load("Group_model.sav")
    matcher = Matcher(nlp.vocab)

    caller_name = None
    app_name = None
    phone_no = None
    time = None
    chat_count = 0

    time_greet = None

    welcome_greeting = None
    welcome_intj = None

    def __init__(self):
        self.add_matchers()             # Add Name Matchers
        self.add_service_entities()     # Add Saloon Services as Entities

        # Load Time Greeting
        self.time_greet = TimeHelper().getCurrentGreeting()

    def preprocess(self,text):
        doc = self.nlp(text)
        # [t.lemma_ for t in doc if not t.is_stop if t.lemma_.isalpha()]
        # [t.text.title() if t.pos_ == 'ADJ' else t.text for t in doc]
        lemma_list = [t.lemma_ for t in doc if t.pos_ != 'PUNCT']
        return self.listToString(lemma_list)
    
    def remove_names(self,text):
        if self.caller_name != None:
            new_set = text.lower().replace(self.caller_name.lower(), 'username')
            return new_set
        else:
            return text
    
    def remove_greetings(self,text):
        if self.welcome_greeting != None:
            new_set = text.lower().replace(self.welcome_greeting.lower(), 'usergreeting')
            return new_set
        else:
            return text

    def remove_welcomeINTJ(self,text):
        if self.welcome_intj != None:
            new_set = text.lower().replace(self.welcome_intj.lower(), 'interjection')
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
    
    def remove_unwanted_text(self,text):
        removed_name = self.remove_names(text)
        removed_greeting = self.remove_greetings(removed_name)
        removed_welcomeINTJ = self.remove_welcomeINTJ(removed_greeting)
        return removed_welcomeINTJ

    def check_name(self,doc):
        self.matches = self.matcher(doc)
        
        if len(self.matches) != 0:
            for match_id, start, end in self.matches:   
                string_id = self.nlp.vocab.strings[match_id]
                if string_id == "Caller_Name1":
                    self.caller_name = str(doc[start+1:end]).title()
                    break
                elif string_id == "Caller_Name2":
                    self.caller_name = str(doc[start]).title()
                    break
            print('Chat Person name is :- ', self.caller_name)
        else:
            print('Chat Person name not detected!')

    def check_greeting(self,doc):
        self.matches = self.matcher(doc)
        
        if len(self.matches) != 0:
            for match_id, start, end  in self.matches:   
                string_id = self.nlp.vocab.strings[match_id]
                if string_id == "Greeting1" or string_id == "Greeting2" or string_id == "Greeting3" or string_id == "Greeting4":
                    self.is_welcome_greeting = True
                    self.welcome_greeting = str(doc[start:end])
                    break
                else:
                    print('No Greeting Detected!')
        else:
            print('No Greeting Detected!')
            pass

    def check_intj(self,doc):
        self.matches = self.matcher(doc)

        if len(self.matches) != 0:
            for match_id, start, end  in self.matches:   
                string_id = self.nlp.vocab.strings[match_id]
                if string_id == "INTJ1":
                    self.welcome_intj = str(doc[start:end])
                    print('Interjection Detected!')
                else:
                    break
        else:
            pass
    
    def check_services(self,doc):
        
        for ent in doc.ents:
            if ent.label_ == "GEN_SERVICE":
                print(ent.text, ent.start_char, ent.end_char, ent.label_)
            else:
                print('No Entities')
                pass

    ############################# Start of - Matchers , Entity Add #############################

    def add_matchers(self):
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
        greeting_pattern_4 = [{'LOWER': 'good', 'OP': '*'},
                                {'LOWER': 'noon'}]
        
        # Detect Interjection
        intj1 = [{'POS': 'INTJ'}]
        
        

        self.matcher.add("Caller_Name1", None, name_pattern_1)
        self.matcher.add("Caller_Name2", None, name_pattern_2)
        self.matcher.add("Greeting1", None, greeting_pattern_1)
        self.matcher.add("Greeting2", None, greeting_pattern_2)
        self.matcher.add("Greeting3", None, greeting_pattern_3)
        self.matcher.add("Greeting3", None, greeting_pattern_4)
        self.matcher.add("INTJ1", None, intj1)

    def add_service_entities(self):
        all_service_data = pd.read_excel('Services.xlsx', header=0) 
        service_data = all_service_data[all_service_data['Type'] == 1]
        service_list = service_data['Service'].tolist()

        service_entity = Entity(keywords_list=service_list, label='GEN_SERVICE')
        self.nlp.add_pipe(service_entity, last=True)

    ############################# End of - Matchers , Entity Add #############################


    ############################# Start of - Chat Helper #############################

    def get_general_greeting1(self):
        greet_data = pd.read_excel('GeneralGreeting.xlsx', header=0) 
        reply_chat = greet_data[greet_data['Type'] == 1]
        item = random.randint(len(reply_chat))
        reply_chat = reply_chat['Chat'][item]

        if self.caller_name != None:
            reply_chat =  reply_chat.replace("USER", self.caller_name)

        if self.time_greet != None:
            reply_chat = reply_chat.replace("GREETING", self.time_greet)
        
        return reply_chat
    
    def get_chat_response(self,text):
        doc = self.nlp(text)
        self.check_name(doc)
        self.check_greeting(doc)
        self.check_intj(doc)
        self.check_services(doc)

        proc_text = self.remove_unwanted_text(text)
        print('Test ||| ', self.preprocess(proc_text))
        pred_group = self.loaded_model.predict([self.preprocess(proc_text)])
        print("Pred Group is ", pred_group)
        if pred_group == 1:
            return self.get_general_greeting1()

    ############################# End of - Chat Helper #############################


class ModelHelper:

    nlp = spacy.load('en_core_web_sm', disable=["ner","textcat"])
    matcher = Matcher(nlp.vocab)

    caller_name = None
    app_name = None
    phone_no = None
    time = None
    chat_count = 0

    welcome_greeting = None
    welcome_intj = None

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
        greeting_pattern_4 = [{'LOWER': 'good', 'OP': '*'},
                                {'LOWER': 'noon'}]
        
        # Detect Interjection
        intj1 = [{'POS': 'INTJ'}]
        
        

        self.matcher.add("Caller_Name1", None, name_pattern_1)
        self.matcher.add("Caller_Name2", None, name_pattern_2)
        self.matcher.add("Greeting1", None, greeting_pattern_1)
        self.matcher.add("Greeting2", None, greeting_pattern_2)
        self.matcher.add("Greeting3", None, greeting_pattern_3)
        self.matcher.add("Greeting3", None, greeting_pattern_4)
        self.matcher.add("INTJ1", None, intj1)

    def preprocess(self,text):
        doc = self.nlp(text)
        # [t.lemma_ for t in doc if not t.is_stop if t.lemma_.isalpha()]
        # [t.text.title() if t.pos_ == 'ADJ' else t.text for t in doc]
        lemma_list = [t.lemma_ for t in doc if t.pos_ != 'PUNCT']
        return self.listToString(lemma_list)
    
    def preprocess_givendata(self,text):
        self.check_name(text)
        self.check_greeting(text)
        self.check_intj(text)

        proc_text = self.remove_unwanted_text(text)
        self.caller_name = None
        self.welcome_greeting = None
        self.welcome_intj = None
        return self.preprocess(proc_text)
    
    def remove_names(self,text):
        if self.caller_name != None:
            new_set = text.lower().replace(self.caller_name.lower(), 'username')
            return new_set
        else:
            return text
    
    def remove_greetings(self,text):
        if self.welcome_greeting != None:
            new_set = text.lower().replace(self.welcome_greeting.lower(), 'usergreeting')
            return new_set
        else:
            return text

    def remove_welcomeINTJ(self,text):
        if self.welcome_intj != None:
            new_set = text.lower().replace(self.welcome_intj.lower(), 'interjection')
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
    
    def remove_unwanted_text(self,text):
        removed_name = self.remove_names(text)
        removed_greeting = self.remove_greetings(removed_name)
        removed_welcomeINTJ = self.remove_welcomeINTJ(removed_greeting)
        return removed_welcomeINTJ

    def check_name(self,text):
        doc = self.nlp(text)
        self.matches = self.matcher(doc)
        
        if len(self.matches) != 0:
            for match_id, start, end in self.matches:   
                string_id = self.nlp.vocab.strings[match_id]
                if string_id == "Caller_Name1":
                    self.caller_name = str(doc[start+1:end]).title()
                    break
                elif string_id == "Caller_Name2":
                    self.caller_name = str(doc[start]).title()
                    break
        else:
            pass

    def check_greeting(self,text):
        doc = self.nlp(text)
        self.matches = self.matcher(doc)
        
        if len(self.matches) != 0:
            for match_id, start, end  in self.matches:   
                string_id = self.nlp.vocab.strings[match_id]
                if string_id == "Greeting1" or string_id == "Greeting2" or string_id == "Greeting3" or string_id == "Greeting4":
                    self.is_welcome_greeting = True
                    self.welcome_greeting = str(doc[start:end])
                    break
                else:
                    pass
        else:
            pass

    def check_intj(self,text):
        doc = self.nlp(text)
        self.matches = self.matcher(doc)

        if len(self.matches) != 0:
            for match_id, start, end  in self.matches:   
                string_id = self.nlp.vocab.strings[match_id]
                if string_id == "INTJ1":
                    self.welcome_intj = str(doc[start:end])
                else:
                    break
        else:
            pass
    



    # def check_matchers(self,text):
    #     doc = self.nlp(text)
    #     self.matches = self.matcher(doc)

    #     text_list = [t.text for t in doc]

    #     if len(self.matches) != 0:
    #         for match_id, start, end  in self.matches:   
    #             string_id = self.nlp.vocab.strings[match_id]
    #             if string_id == "Caller_Name1":
    #                 self.caller_name = str(doc[start+1:end]).title()
    #                 text_list[start+1:end] = 'username'
    #             elif string_id == "Caller_Name2":
    #                 self.caller_name = str(doc[start]).title()
    #                 text_list[start] = 'username'
    #             elif string_id == "Greeting1" or string_id == "Greeting2" or string_id == "Greeting3" or string_id == "Greeting4":
    #                 self.is_welcome_greeting = True
    #                 self.welcome_greeting = str(doc[start:end])
    #                 text_list[start:end] = 'usergreeting'
    #             elif string_id == "INTJ1":
    #                 self.welcome_intj = str(doc[start:end])
    #                 text_list[start:end] = 'interjection'
    #                 print('Test ', text)
    #                 print('Interjection Detected!')
    #     else:
    #         print('No Matches Detected!')
    #         pass
        
    #     return self.listToString(text_list)

class TimeHelper:

    def getCurrentGreeting(self):
        now = datetime.datetime.now()
        # now.strftime("%Y-%b-%d, %A %I:%M:%S")
        current_hour = int(now.strftime("%H"))
        print('Time ', current_hour)

        if current_hour >= 0 and current_hour < 12:
            return "Good Morning"
        elif current_hour >= 12 and current_hour < 16:
            return "Good Afternoon"
        else:
            return "Good Evening"
    
    def getFinalGreeting(self):
        now = datetime.datetime.now()
        current_hour = int(now.strftime("%H"))

        if current_hour >= 0 and current_hour < 12:
            return "Have a nice day"
        elif current_hour >= 12 and current_hour < 16:
            return "Have a nice day"
        else:
            return "Good Night!"