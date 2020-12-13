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

    nlp = None #spacy.load('en_core_web_sm')
    # load the model from disk
    loaded_model = joblib.load("Group_model.sav")
    loaded_acc_rej_model = joblib.load("Acc_Rej_model.sav")
    matcher = None #Matcher(nlp.vocab)

    # Available Services
    gen_service_tags_df = None
    service_info = None
    reply_data = None
    appointment_reply = None

    caller_name = None
    phone_no = None
    re_name = None
    re_service = None
    re_ser_code = None
    re_time = None
    re_date = None

    re_name_checked = False
    re_status = 0 # 0 - NA, 1 - In progress(Pending Conformation), 2 - Cancel, 3 - Completed! (Enum can be used also)
    # re_date_checked = False
    # re_service_checked = False
    # re_phone_no_checked = False

    checked_services_codes = []
    checked_services_tags = []
    is_newProd_check = False
    chat_count = 0

    time_greet = None

    welcome_greeting = None
    welcome_intj = None

    cluster3_words = None

    is_make_reser_mode = False
    is_change_reser_mode = False
    is_remove_reser_mode = False


    def __init__(self):
        self.nlp = spacy.load('en_core_web_sm')
        self.matcher = Matcher(self.nlp.vocab)
        self.gen_service_tags_df = pd.read_excel('Services_tags.xlsx', header=0)
        self.service_info = pd.read_excel('Service_details.xlsx', header=0)
        self.reply_data = pd.read_excel('ReplyChat.xlsx', header=0)
        self.appointment_reply = pd.read_excel('Appointment_Reply.xlsx', header=0)

        self.load_cluster3_words()

        self.add_matchers()             # Add Name Matchers
        self.add_service_entities()     # Add Saloon Services as Entities

        # Load Time Greeting
        self.time_greet = TimeHelper().getCurrentGreeting()



############################# --- Text Preprocess --- #############################

    def preprocess(self,text):
        doc = self.nlp(text)
        # [t.lemma_ for t in doc if not t.is_stop if t.lemma_.isalpha()]
        # [t.text.title() if t.pos_ == 'ADJ' else t.text for t in doc]
        lemma_list = [t.text for t in doc if t.pos_ != 'PUNCT']
        return self.listToString(lemma_list)
    
    def preprocess_givendata(self,text):
        doc = self.nlp(text)
        self.check_all_elements(doc)
        # self.check_name(doc)
        # self.check_greeting(doc)
        # self.check_intj(doc)
        # self.check_services(doc)

        proc_text = self.remove_unwanted_text(text)
        self.caller_name = None
        self.re_name = None
        self.phone_no = None
        self.time = None
        self.checked_services_codes = []
        self.checked_services_tags = []

        self.time_greet = None

        self.welcome_greeting = None
        self.welcome_intj = None

        return self.preprocess(proc_text)
    
    def preprocess_cluster_data(self,text):   
        doc = self.nlp(text)
        lemma_list = [t.lemma_ for t in doc if t.pos_ != 'PUNCT' if not t.is_stop if t.pos_ == 'NOUN' or t.pos_ == 'VERB' or t.pos_ == 'ADJ']
        return lemma_list

    def load_cluster3_words(self):
        chat_data = pd.read_excel('UserChatData.xlsx', header=0) 
        clust_3_df = chat_data[chat_data['Group'] == 3]
        ret_list = []
        
        for row in clust_3_df.itertuples():
            text_list = self.preprocess_cluster_data(row[1])
            ret_list.extend(text_list)
        self.cluster3_words = list(dict.fromkeys(ret_list))

    def is_text_cluster3(self,text):
        text_word_list = self.preprocess_cluster_data(text)
        if len(list(set(text_word_list) - set(self.cluster3_words))) > 0:
            return False
        else:
            return True

############################# --- Text word removers --- #############################

    def remove_names(self,text):
        if self.caller_name != None:
            new_set = text.lower().replace(self.caller_name.lower(), 'username')
            return new_set
        else:
            return text

    def remove_greetings(self,text):
        if self.welcome_greeting != None:
            new_set = text.lower().replace(self.welcome_greeting.lower(), 'usergreet')
            return new_set
        else:
            return text

    def remove_welcomeINTJ(self,text):
        if self.welcome_intj != None:
            new_set = text.lower().replace(self.welcome_intj.lower(), 'interjection')
            return new_set
        else:
            return text

    def remove_services(self,text):
        ret_text = text
        for serv in self.checked_services_tags:
            ret_text = ret_text.lower().replace(serv, 'gservice')
        return ret_text
    
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
        removed_services = self.remove_services(removed_welcomeINTJ)
        return removed_services



############################# --- Word Checkers --- #############################

    def check_name(self,doc):
        self.matches = self.matcher(doc)
        
        if len(self.matches) != 0:
            for match_id, start, end in self.matches:   
                string_id = self.nlp.vocab.strings[match_id]
                if string_id == "Caller_Name1":
                    self.caller_name = str(doc[start+1:end]).title()
                    if self.caller_name[len(self.caller_name) - 1] == ".":
                        self.caller_name = self.caller_name[0:len(self.caller_name) - 1]
                    self.re_name = self.caller_name
                    break
                elif string_id == "Caller_Name2":
                    self.caller_name = str(doc[start]).title()
                    if self.caller_name[len(self.caller_name) - 1] == ".":
                        self.caller_name = self.caller_name[0:len(self.caller_name) - 1]
                    self.re_name = self.caller_name
                    break
        else:
            pass

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
                    pass
        else:
            pass

    def check_intj(self,doc):
        self.matches = self.matcher(doc)

        if len(self.matches) != 0:
            for match_id, start, end  in self.matches:   
                string_id = self.nlp.vocab.strings[match_id]
                if string_id == "INTJ1":
                    self.welcome_intj = str(doc[start:end])
                else:
                    pass
        else:
            pass
    
    def check_services(self,doc):
        for ent in doc.ents:
            if ent.label_ == "GEN_SERVICE":
                ind_val = self.gen_service_tags_df.loc[self.gen_service_tags_df['Tag'] == ent.text.lower()]
                self.checked_services_codes.append(ind_val['Code'].values[0])
                self.checked_services_tags.append(ent.text.lower())
                self.is_newProd_check = True
            else:
                pass

    def check_client_name(self,text):
        doc = self.nlp(text)
        name_list = [t.text for t in doc if t.pos_ == 'PROPN' if not t.is_stop]
        
        if len(name_list) != 0:
            det_name = self.listToString(name_list)
            print("Name detected:- ", det_name)
            self.re_name = det_name.title()
        else:
            print("Name not detected!")
            self.re_name = ""

    def has_user_feedback(self,text):
        # Add extra user feedback words of bag
        extra_feedback_words = ['indeed', 'correct', 'not', 'absolutely', 'definitely', 'right', 'accurate', 'wrong', 'inaccurate']
        have_user_feedback = False
        have_name = False
        n_doc = self.nlp(text)
        name_list = [t.text for t in n_doc if t.pos_ == 'PROPN' if not t.is_stop]

        for token in n_doc:
            if token.pos_ == "INTJ":
                have_user_feedback = True
                break
            elif token.text.lower() in extra_feedback_words:
                have_user_feedback = True
                break
        if len(name_list) != 0:
            have_name = True
        return have_user_feedback, have_name

    def detect_service_date_phone(self,text):
        doc = self.nlp(text)
        for ent in doc.ents:
            if ent.label_ == "GEN_SERVICE":
                print('Service Found!')
                ind_val = self.gen_service_tags_df.loc[self.gen_service_tags_df['Tag'] == ent.text.lower()]
                if self.re_service != None:
                    self.re_service = self.re_service.replace(' & ', ' , ')
                    self.re_ser_code = self.re_ser_code.replace(' & ', ' , ')

                    self.re_service = self.re_service + " & " + ent.text.title()
                    self.re_ser_code = self.re_ser_code + " & " + ind_val['Code'].values[0]
                else:
                    self.re_service = ent.text.title()
                    self.re_ser_code = ind_val['Code'].values[0]
            elif ent.label_ == "DATE" and not ent.text.isnumeric():
                print('Reservation Date found!')
                if self.re_date != None:
                    self.re_date = self.re_date .replace(' & ', ' , ')
                    self.re_date = self.re_date + " & " + ent.text
                else:
                    self.re_date = ent.text
            elif ent.label_ == "CARDINAL" or ent.label_ == "DATE" and ent.text.isnumeric():
                print('Reservation Phone number found!')
                if self.phone_no != None:
                    self.phone_no = self.phone_no .replace(' & ', ' , ')
                    self.phone_no = self.phone_no + " & " + ent.text
                else:
                    self.phone_no = ent.text

    def check_all_elements(self,doc):
        self.check_name(doc)
        self.check_greeting(doc)
        self.check_intj(doc)
        self.check_services(doc)


############################# --- Matchers , Entity Add --- #############################

    def add_matchers(self):
        # Add name Detect Matcher
        name_pattern_1 = [{'POS': 'AUX', 'OP': '+'},
                            {'POS': 'PROPN', 'OP': '+'},
                            {'POS': 'PROPN', 'OP': '!', 'DEP': 'compound'}]
        # name_pattern_1 = [{'POS': 'AUX', 'OP': '*'},
        #                     {'POS': 'PROPN', 'OP': '+'},
        #                     {'POS': 'PROPN', 'OP': '*'}]
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
        service_data = self.gen_service_tags_df[self.gen_service_tags_df['Type'] == 1]
        service_list = service_data['Tag'].tolist()

        service_entity = Entity(keywords_list=service_list, label='GEN_SERVICE')
        self.nlp.add_pipe(service_entity, last=True)


############################# --- Add,Change,Remove Reservation --- #############################


    def check_appointment_name(self,re_type):
        # re_type = 5  => Make Reservation
        # re_type = 6  => Change Reservation
        # re_type = 7  => Cancel Reservation

        group_df = self.appointment_reply[self.appointment_reply['Group'] == re_type]
        if self.re_name_checked == False:
            if self.re_name == self.caller_name and self.re_name != None:
                # Check we could make appointment for chat person name
                reply_chat = group_df[group_df['Code'] == 'NAME_CHECK_P']
                item = random.randint(len(reply_chat))
                reply_chat = reply_chat['Chat'].tolist()[item]
                return False, reply_chat.replace("USER", self.re_name)
            elif self.re_name == None:
                # self.re_name = "" when not detected appointment person name
                reply_chat = group_df[group_df['Code'] == 'NAME_CHECK_NA']
                item = random.randint(len(reply_chat))
                return False, reply_chat['Chat'].tolist()[item]
            elif self.re_name == "":
                # self.re_name = "" when not detected appointment person name
                reply_chat = group_df[group_df['Code'] == 'NAME_CHECK_ERROR']
                item = random.randint(len(reply_chat))
                return False, reply_chat['Chat'].tolist()[item]
            else:
                # Get the reservation name
                reply_chat = group_df[group_df['Code'] == 'NAME_CHECK_N']
                item = random.randint(len(reply_chat))
                reply_chat = reply_chat['Chat'].tolist()[item]
                return True, reply_chat.replace("USER", self.re_name)
        else:
            # Name Confirmed!
            return True, "Name Confirmed!"
        

############################# --- Chat Helper --- #############################

    def get_general_greeting(self):
        #greet_data = pd.read_excel('ReplyChat.xlsx', header=0) 

        reply_chat = self.reply_data[self.reply_data['Type'] == 1]
        item = random.randint(len(reply_chat))
        reply_chat = reply_chat['Chat'].tolist()[item]

        if self.caller_name != None:
            reply_chat =  reply_chat.replace("USER", self.caller_name)
        else:
            reply_chat =  reply_chat.replace("USER", "")

        if self.time_greet != None:
            reply_chat = reply_chat.replace("GREETING", self.time_greet)
        else:
            reply_chat = reply_chat.replace("GREETING", "")
        
        return reply_chat

    def get_checkback_greeting(self):
        #greet_data = pd.read_excel('ReplyChat.xlsx', header=0) 

        reply_chat = self.reply_data[self.reply_data['Type'] == 2]
        item = random.randint(len(reply_chat))
        reply_chat = reply_chat['Chat'].tolist()[item]

        if self.caller_name != None:
            reply_chat =  reply_chat.replace("USER", self.caller_name)
        else:
            reply_chat =  reply_chat.replace("USER", "")

        if self.time_greet != None:
            reply_chat = reply_chat.replace("GREETING", self.time_greet)
        else:
            reply_chat = reply_chat.replace("GREETING", "")
        
        return reply_chat

    def get_service_information(self):

        if self.is_newProd_check:
            ind_val = self.service_info.loc[self.service_info['Code'] == self.checked_services_codes[0]]
            self.is_newProd_check = False
            return ind_val['Description'].values[0]
        else:
            ind_val = self.service_info.loc[self.service_info['Code'] == 'ERROR']
            return ind_val['Description'].values[0]
    
    def get_general_information(self,text):

        if self.is_text_cluster3(text):
            ind_val = self.service_info.loc[self.service_info['Code'] == 'G000']
            return ind_val['Description'].values[0]
        else:
            ind_val = self.service_info.loc[self.service_info['Code'] == 'ERROR']
            return ind_val['Description'].values[0]
    
    def get_general_reser_reply(self,re_type):
        # re_type = 5  => Make Reservation
        # re_type = 6  => Change Reservation
        # re_type = 7  => Cancel Reservation

        reply_chat = self.reply_data[self.reply_data['Type'] == re_type]
        item = random.randint(len(reply_chat))
        reply_chat = reply_chat['Chat'].tolist()[item]

        if self.caller_name != None:
            reply_chat =  reply_chat.replace("USER", self.caller_name)
        else:
            reply_chat =  reply_chat.replace("USER", "")
        
        return reply_chat

    def get_reservation_confirmation(self):
        reply_chat = self.reply_data[self.reply_data['Type'] == 52]
        item = random.randint(len(reply_chat))
        reply_chat = reply_chat['Chat'].tolist()[item]

        reply_chat = reply_chat.replace("USER", self.re_name)
        reply_chat = reply_chat.replace("GSERVICE", self.re_service)
        reply_chat = reply_chat.replace("APPDATE", self.re_date)
        reply_chat = reply_chat.replace("PHONUM", self.phone_no)

        if self.caller_name != None:
            reply_chat =  reply_chat.replace("CALLER", self.caller_name)
        else:
            reply_chat =  reply_chat.replace("CALLER", "")
        
        return reply_chat

    def get_chat_response(self,text):
        doc = self.nlp(text)
        self.check_all_elements(doc)
        # self.check_name(doc)
        # self.check_greeting(doc)
        # self.check_intj(doc)
        # self.check_services(doc)

        if self.is_make_reser_mode == False and self.is_change_reser_mode == False and self.is_remove_reser_mode == False:
            proc_text = self.remove_unwanted_text(text)
            print('Test ||| ', self.preprocess(proc_text))
            pred_group = self.loaded_model.predict([self.preprocess(proc_text)])[0]
            print("Pred Group is ", pred_group)
            print('Chat Person Name :- ', self.caller_name)
            print('Greeting :- ', self.welcome_greeting)
            print('Service Codes :- ', self.checked_services_codes)
            print('Service Tags :- ', self.checked_services_tags)

            if pred_group == 1:
                return self.get_general_greeting()
            elif pred_group == 2:
                return self.get_checkback_greeting()
            elif pred_group == 3:
                return self.get_general_information(proc_text)
            elif pred_group == 4:
                return self.get_service_information()
            elif pred_group == 5:
                reservation_text = self.get_general_reser_reply(5) + self.check_appointment_name(5)[1]
                self.is_make_reser_mode = True
                return reservation_text
            elif pred_group == 6:
                reservation_text = self.get_general_reser_reply(6) + self.check_appointment_name(6)[1]
                self.is_change_reser_mode = True
                return reservation_text
            elif pred_group == 7:
                reservation_text = self.get_general_reser_reply(6) + self.check_appointment_name(6)[1]
                self.is_remove_reser_mode = True
                return reservation_text
        elif self.is_make_reser_mode == True:
            if self.re_name_checked == False:
                has_feedback, has_name = self.has_user_feedback(text)
                if has_feedback and has_name:
                    self.check_client_name(text)
                    pred_user_feedback = self.loaded_acc_rej_model.predict([text])[0]
                    if pred_user_feedback == "ACCEPT":
                        self.re_name_checked = True
                        reply_chat = self.reply_data[self.reply_data['Type'] == 51]
                        item = random.randint(len(reply_chat))
                        return reply_chat['Chat'].tolist()[item]
                    else:
                        #self.check_client_name(text)
                        return self.check_appointment_name(5)[1]
                elif has_feedback:
                    #self.check_client_name(text)
                    pred_user_feedback = self.loaded_acc_rej_model.predict([text])[0]
                    if pred_user_feedback == "ACCEPT":
                        self.re_name_checked = True
                        reply_chat = self.reply_data[self.reply_data['Type'] == 51]
                        item = random.randint(len(reply_chat))
                        return reply_chat['Chat'].tolist()[item]
                    else:
                        self.check_client_name(text)
                        return self.check_appointment_name(5)[1]
                else:
                    self.check_client_name(text)
                    return self.check_appointment_name(5)[1]
            else:
                self.detect_service_date_phone(text)
                if self.phone_no != None and self.re_ser_code != None and self.re_date != None:
                    if self.re_status == 0:
                        self.re_status = 1
                        return self.get_reservation_confirmation()
                    elif self.re_status == 1:
                        pred_user_feedback = self.loaded_acc_rej_model.predict([text])[0]
                        if pred_user_feedback == "ACCEPT":
                            self.re_status = 4
                            self.is_make_reser_mode = False
                            reply_chat = self.reply_data[self.reply_data['Type'] == 53]
                            item = random.randint(len(reply_chat))
                            reply_chat = reply_chat['Chat'].tolist()[item]
                            if self.caller_name != None:
                                reply_chat =  reply_chat.replace("CALLER", self.caller_name)
                            else:
                                reply_chat =  reply_chat.replace(" CALLER", "")
                            return reply_chat
                        else:
                            self.re_status = 0
                            self.phone_no = None
                            self.re_service = None
                            self.re_ser_code = None
                            self.re_date = None

                            reply_chat = self.reply_data[self.reply_data['Type'] == 54]
                            item = random.randint(len(reply_chat))
                            reply_chat = reply_chat['Chat'].tolist()[item]
                            if self.caller_name != None:
                                reply_chat =  reply_chat.replace("CALLER", self.caller_name)
                            else:
                                reply_chat =  reply_chat.replace(" CALLER", "")
                            return reply_chat
                elif self.phone_no == None:
                    self.phone_no = None
                    reply_chat = self.reply_data[self.reply_data['Type'] == 55]
                    item = random.randint(len(reply_chat))
                    return reply_chat['Chat'].tolist()[item]
                elif self.re_ser_code == None:
                    self.re_ser_code = None
                    self.re_service = None
                    reply_chat = self.reply_data[self.reply_data['Type'] == 56]
                    item = random.randint(len(reply_chat))
                    return reply_chat['Chat'].tolist()[item]
                elif self.re_date == None:
                    self.re_date = None
                    reply_chat = self.reply_data[self.reply_data['Type'] == 57]
                    item = random.randint(len(reply_chat))
                    return reply_chat['Chat'].tolist()[item]
        elif self.is_change_reser_mode == True:
            pass
        elif self.is_remove_reser_mode == True:
            pass







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



"""
class ModelHelper:

    nlp = spacy.load('en_core_web_sm', disable=["ner","textcat"])
    matcher = Matcher(nlp.vocab)

    caller_name = None
    app_name = None
    phone_no = None
    time = None
    checked_services_codes = []
    checked_services_tags = []
    chat_count = 0

    # Available Services
    gen_service_tags_df = None

    welcome_greeting = None
    welcome_intj = None

    def __init__(self):
        self.gen_service_tags_df = pd.read_excel('Services_tags.xlsx', header=0)

        # Add Name Matchers
        self.add_matchers()             # Add Name Matchers
        self.add_service_entities()     # Add Saloon Services as Entities


    def preprocess(self,text):
        doc = self.nlp(text)
        # [t.lemma_ for t in doc if not t.is_stop if t.lemma_.isalpha()]
        # [t.text.title() if t.pos_ == 'ADJ' else t.text for t in doc]
        lemma_list = [t.text for t in doc if t.pos_ != 'PUNCT']
        return self.listToString(lemma_list)
    
    def preprocess_givendata(self,text):
        doc = self.nlp(text)
        self.check_name(doc)
        self.check_greeting(doc)
        self.check_intj(doc)
        self.check_services(doc)

        proc_text = self.remove_unwanted_text(text)
        self.caller_name = None
        self.welcome_greeting = None
        self.welcome_intj = None
        self.checked_services_codes = []
        self.checked_services_tags = []
        return self.preprocess(proc_text)
    
    def remove_names(self,text):
        if self.caller_name != None:
            new_set = text.lower().replace(self.caller_name.lower(), 'username')
            return new_set
        else:
            return text
    
    def remove_greetings(self,text):
        if self.welcome_greeting != None:
            new_set = text.lower().replace(self.welcome_greeting.lower(), 'usergreet')
            return new_set
        else:
            return text

    def remove_welcomeINTJ(self,text):
        if self.welcome_intj != None:
            new_set = text.lower().replace(self.welcome_intj.lower(), 'interjection')
            return new_set
        else:
            return text
    
    def remove_services(self,text):
        ret_text = text
        for serv in self.checked_services_tags:
            ret_text = ret_text.lower().replace(serv, 'gservice')
        return ret_text

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
        removed_services = self.remove_services(removed_welcomeINTJ)
        return removed_services

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
        else:
            pass

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
                    pass
        else:
            pass

    def check_intj(self,doc):
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
    
    def check_services(self,doc):
        for ent in doc.ents:
            if ent.label_ == "GEN_SERVICE":
                ind_val = self.gen_service_tags_df.loc[self.gen_service_tags_df['Tag'] == ent.text.lower()]
                self.checked_services_codes.append(ind_val['Code'].values[0])
                self.checked_services_tags.append(ent.text.lower())
                print('Service Code', ind_val['Code'].values[0])
                print(ent.text, ent.start_char, ent.end_char, ent.label_)
            else:
                print('No Entities')
                pass

    ############################# Start of - Matchers , Entity Add #############################

    def add_matchers(self):
        # name_pattern_1 = [{'POS': 'AUX', 'OP': '+'},
        #                     {'POS': 'PROPN', 'OP': '+'},
        #                     {'POS': 'PROPN', 'OP': '!', 'DEP': 'compound'}]
        name_pattern_1 = [{'POS': 'AUX', 'OP': '+'},
                            {'POS': 'PROPN', 'OP': '+'}]
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
        service_data = self.gen_service_tags_df[self.gen_service_tags_df['Type'] == 1]
        service_list = service_data['Tag'].tolist()

        service_entity = Entity(keywords_list=service_list, label='GEN_SERVICE')
        self.nlp.add_pipe(service_entity, last=True)

    ############################# End of - Matchers , Entity Add #############################

"""


