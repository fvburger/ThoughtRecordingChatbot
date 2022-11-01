# This files contains your custom actions which can be used to run custom Python code.

# See this guide on how to implement these action:
# https://rasa.com/docs/rasa/custom-actions


# This is a simple example for a custom action which utters "Hello World!"

from typing import Any, Text, Dict, List
from rasa_sdk import Action, Tracker
from rasa_sdk.executor import CollectingDispatcher
from rasa_sdk.events import SlotSet
from rasa_sdk.events import FollowupAction, EventType
from rasa_sdk.forms import FormAction, FormValidationAction
import os
import logging
import keras
import matplotlib.pyplot as plt; plt.rcdefaults()
import numpy as np
import sqlite3
import traceback
import pandas
import random
import re
import string
import contractions
from textblob import TextBlob
import gensim
from gensim.parsing.preprocessing import remove_stopwords
from num2words import num2words
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
import csv
from math import pi
import json
from operator import add
from imgurpython import ImgurClient
import datetime

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # FATAL
logging.getLogger('tensorflow').setLevel(logging.FATAL)

#as taken from mitchell
#function to extract custom data from rasa webchat (in my case only the prolific id)
def extract_metadata_from_tracker(tracker: Tracker):
	events = tracker.current_state()['events']
	user_events = []
	for e in events:
		if e['event'] == 'user':
			user_events.append(e)

	return user_events[-1]['metadata']

class ActionSetInitialSlots(Action):
	def name(self) -> Text:
		return "caction_setsession"

	async def run(
		self,
		dispatcher: CollectingDispatcher,
		tracker: Tracker,
		domain: Dict[Text, Any],) -> List[EventType]:
		
		feedback_types = ["nofb","process","content"]

		#set session
		metadata = extract_metadata_from_tracker(tracker)
		user_id = metadata['userid']
		logging.info("userid: {}".format(user_id))
		reloadsession = False
		try:
			mydb = sqlite3.connect("ANTbot.db")
			mycursor = mydb.cursor()
			logging.info("Connection established, ready to query session data of this participant")
			sqlquery = "SELECT `s1comp`,`condition` FROM Demographics WHERE participantID = ? "
			mycursor.execute(sqlquery,[user_id])
			myresult = mycursor.fetchall()
			logging.info("myresult: {}".format(myresult))
			if not myresult:
				val = True
				tr_counter = 0
			else:
				if myresult[0][0] == 1:
					val = False
					feedback_type = myresult[0][1]
					tr_counter = 2
				elif myresult[0][0] is None:
					val = True
					reloadsession = True
					tr_counter = 0
					feedback_type = myresult[0][1]
				else:
					val = True
					tr_counter = 0
		except:
			logging.info("Error occured while connecting to database. Error Trace: {}".format(traceback.format_exc()))
		finally:
			if mydb:
				mycursor.close()
				mydb.close()
				logging.info("SQL connection is closed")

		if val and not reloadsession:
			moderator_bucket = metadata['modbucket']
			if moderator_bucket not in ["1","2","3"]:
				moderator_bucket = random.choice(["1","2","3"])
			modconds = []
			try:
				mydb= sqlite3.connect("ANTbot.db")
				mycursor = mydb.cursor()
				logging.info("Connection established, ready to query session data of all participants")
				sqlquery = "SELECT `condition`, `modbucket` FROM Demographics"
				mycursor.execute(sqlquery)
				myresult = mycursor.fetchall()
				for res in myresult:
					modconds.append(res)
				if not modconds:
					feedback_type = random.choice(feedback_types)
				else:
					df = pandas.DataFrame(0, columns=['mb1','mb2','mb3'], index=feedback_types)
					for modcond in modconds:
						cond = modcond[0]
						mod = 'mb' + str(modcond[1])
						df.loc[cond,mod] += 1
					index = int(moderator_bucket) - 1
					dist = df.iloc[:,index].values
					lowest_cond = np.argmin(dist)
					logging.info("lowest cond number: {}".format(lowest_cond))
					if lowest_cond > 2 or lowest_cond < 0:
						feedback_type = random.choice(feedback_types)
						logging.info("feedback type was picked at random.")
					else:
						feedback_type = feedback_types[lowest_cond]
				sqlquery2 = "INSERT INTO Demographics (`participantID`, `condition`, `modbucket`) VALUES (?, ?, ?)"
				args = (user_id,feedback_type,moderator_bucket)
				mycursor.execute(sqlquery2,args)
				mydb.commit()
				logging.info("condition: {}".format(feedback_type))
			except:
				logging.info("Error occured while connecting to database or entering data into database. Error Trace: {}".format(traceback.format_exc()))
			finally:
				if mydb:
					mycursor.close()
					mydb.close()
					logging.info("SQL connection is closed")

		return [SlotSet("session1", val), 
		SlotSet("condition", feedback_type),
		SlotSet("participantID", user_id),
		SlotSet("tr_counter", tr_counter)]

class ActionToggleStartDat(Action):
	def name(self) -> Text:
		return "caction_togglestartdat"

	async def run(
		self,
		dispatcher: CollectingDispatcher,
		tracker: Tracker,
		domain: Dict[Text, Any],) -> List[EventType]:

		startdat = tracker.get_slot("start_dat")

		return [SlotSet("start_dat",not startdat)]

class ActionChooseScenario(Action):
	def name(self) -> Text:
		return "caction_choosescenario"

	async def run(
		self,
		dispatcher: CollectingDispatcher,
		tracker: Tracker,
		domain: Dict[Text, Any],) -> List[EventType]:

		session = tracker.get_slot("session1")
		scen1 = tracker.get_slot("scen1")
		scen2 = tracker.get_slot("scen2")
		counter = tracker.get_slot("tr_counter")

		scenarios = pandas.read_csv("supp_materials/scenarios.txt", sep='\t', index_col=False)

		indx_interpersonal = [0,1,2,3,4]
		indx_achievement = [5,6,7,8,9]


		#pick the first scenario at random
		if session:
			if counter == 0:
				scen1 = random.choice(indx_interpersonal + indx_achievement)
				utterance = "Scenario: " + scenarios.iat[scen1,1]
			elif counter == 1:
				if scen1 in indx_interpersonal:
					scen2 = random.choice(indx_achievement)
					utterance = "Scenario: " + scenarios.iat[scen2,1]
				elif scen1 in indx_achievement:
					scen2 = random.choice(indx_interpersonal)
					utterance = "Scenario: " + scenarios.iat[scen2,1]
				else:
					scen2 = 10
					utterance = "Oops, technical problem! Please contact the researcher."
			else:
				utterance = "Oops, technical problem! Please contact the researcher."
		elif not session:
			utterance = "Your turn to describe a recent situation from your life that caused you to feel a strong, negative emotion!"
		else:
			utterance = "Oops, technical problem! Please contact the researcher."

		dispatcher.utter_message(text=utterance)

		return [SlotSet("scen1",scen1),
		SlotSet("scen2",scen2)]

class ActionIncreaseTRCounter(Action):
	def name(self) -> Text:
		return "caction_increaseTRcounter"

	async def run(
		self,
		dispatcher: CollectingDispatcher,
		tracker: Tracker,
		domain: Dict[Text, Any],) -> List[EventType]:

		counter = tracker.get_slot("tr_counter") + 1

		return [SlotSet("tr_counter",counter),SlotSet("dbsave",None)]

class ActionIncreaseSTEPCounter(Action):
	def name(self) -> Text:
		return "caction_increaseSTEPcounter"

	async def run(
		self,
		dispatcher: CollectingDispatcher,
		tracker: Tracker,
		domain: Dict[Text, Any],) -> List[EventType]:

		counter = tracker.get_slot("step_counter") + 1

		return [SlotSet("step_counter",counter)]

class ActionShowExample(Action):
	def name(self) -> Text:
		return "caction_showexampletr"

	async def run(
		self,
		dispatcher: CollectingDispatcher,
		tracker: Tracker,
		domain: Dict[Text, Any],) -> List[EventType]:

		dispatcher.utter_message(image='https://imgur.com/LNcElle.png')

		return []

class ActionSaveThought(Action):
	def name(self) -> Text:
		return "caction_savethought"

	async def run(
		self,
		dispatcher: CollectingDispatcher,
		tracker: Tracker,
		domain: Dict[Text, Any],) -> List[EventType]:

		at = tracker.get_slot("eautomaticthought")
		anxstep = tracker.get_slot("anxstep")
		sadangstep = tracker.get_slot("sadangstep")
		thoughts = tracker.get_slot("thoughts")
		emotype = tracker.get_slot("demotype")

		if at is not None:
			val = at
			if emotype != "happy":
				dat = True
			else:
				dat = False
		elif anxstep is not None:
			val = anxstep
			dat = False
		else:
			val = sadangstep
			dat = False

		thoughts.append(val)

		if emotype == "sad" or emotype == "angry":
			emotype_internal = "sadang"
		else:
			emotype_internal = emotype

		return [SlotSet("previous_thought",val),
		SlotSet("eautomaticthought",None),
		SlotSet("anxstep",None),
		SlotSet("sadangstep",None),
		SlotSet("thoughts",thoughts),
		SlotSet("start_dat",dat),
		SlotSet("emotype_internal",emotype_internal)]


class ActionPrepareFeedback(Action):
	def name(self) -> Text:
		return "caction_prepare_feedback"

	async def run(
		self,
		dispatcher: CollectingDispatcher,
		tracker: Tracker,
		domain: Dict[Text, Any],) -> List[EventType]:

		partID = tracker.get_slot("participantID")
		cond = tracker.get_slot("condition")
		session = tracker.get_slot("session1")
		tr_indx = tracker.get_slot("tr_counter")
		allthoughts = tracker.get_slot("thoughts")
		emotype = tracker.get_slot("emotype_internal")
		
		if emotype != "happy":
			schemas = ["Attachment","Competence","Global self-evaluation","Health","Power and control","Meta-Cognition","Other people","Hopelessness","Other's views on self"] 
			stephistory =[]
			schemahistory =[]
			#mysql database credentials for when deployed on server!
			try:
				mydb= sqlite3.connect("ANTbot.db")
				mycursor = mydb.cursor()
				logging.info("Connection established, ready to query previous data")
				if not tr_indx==1:
					if cond != "nofb":
						sqlquery = "select thoughtRecord, DATsteps from CoreData where participantID = ? "
						mycursor.execute(sqlquery, [partID])
						myresult = mycursor.fetchall()
						for res in myresult:
							stephistory.append(res)
	#					logging.info("stephistory: ", stephistory)
					if cond == "content":
						sqlquery = "select schemadist from CoreData where participantID = ? "
						mycursor.execute(sqlquery, [partID])
						myresult = mycursor.fetchall()
						for res in myresult:
							res = json.loads(res[0])
							res = res["schemas"]
							if not res:
								res = [0,0,0,0,0,0,0,0,0]
							schemahistory.append(res)
	#					logging.info("schemahistory: ",schemahistory)
			except:
				logging.info("Error occured while connecting to database or entering data into database. Error Trace: {}".format(traceback.format_exc()))
			finally:
				if mydb:
					mycursor.close()
					mydb.close()
					logging.info("MySQL connection is closed")

		#imgur api credentials
		client_id = '10e7dbb4a7cda4e'
		client_secret = '828a7e5e58d3d6a386fad165e8a6d36adf323911'
		access_token = '9aa73631a917f5018700b0a4d6badab58985ced5'
		refresh_token = '73b342e21f8e653aeb6e28c547ad3ea849619f69'

		client = ImgurClient(client_id, client_secret, refresh_token)
		client.set_user_auth(access_token, refresh_token)

		schema = None
		c_schemadist = []
		schema_list = []
		pimage_link = None
		cimage_link = None

		#convert schema_history to np.matrix from list of tuples via list of lists
		if emotype != "happy":
			if cond != "nofb":
				#count number of thoughts
				number_of_thoughts = len(allthoughts) - 1 #subtract 1 for automatic thought
				stephistory.append((tr_indx,number_of_thoughts))
#				logging.info(stephistory)
				stephistory = pandas.DataFrame(stephistory, columns=["TR","StepCount"])
				dfcomp = pandas.read_csv("supp_materials/trs_per_depth.txt", sep='\t', index_col=False)
				fig = plt.figure()
				ax1 = fig.add_subplot(111)
				ylength = stephistory["StepCount"].max() + 3
				ax1.plot("TR","StepCount",data=stephistory,marker="o",linewidth=2, markersize=12)
				ax1.set_xlabel("Thought record number",labelpad=20)
				ax1.set_ylabel("Number of completed steps",labelpad=20)
				ax1.set_xticks(np.arange(1, tr_indx+3, step=1))
				ax1.set_yticks(np.arange(1, ylength, step=1))
				ax2 = plt.twinx()
				percdata=dfcomp['cumperc'].astype("int").tolist()[0:ylength-1]
				percdata = [str(i)+'%' for i in percdata] 
				ax1.hlines(y=np.arange(1,ylength,step=1), xmin=[0]*(ylength-1), xmax=[tr_indx+3]*(ylength-1), colors='gray', linestyles='--', lw=1)
				ax2.set_ylim(ax1.get_ylim())
				ax2.set_yticks(np.arange(1, ylength, step=1))
				ax2.yaxis.set_ticklabels(percdata) # change the ticks' names to x
				ax2.set_ylabel("Previous study",rotation=90, labelpad=20)
				plt.tight_layout()
				filename_process = str(partID)+ "_processfb_" + "tr" + str(tr_indx)
				imagepath_process = "fbplots/" + filename_process + '.png'
				plt.savefig(imagepath_process)
				config = {'title': filename_process} 
				pimage = client.upload_from_path(imagepath_process, config=config,anon=False)
				pimage_link = pimage['link']
				plt.close()
			if cond == "content":  
				prepthoughts=[]
				#preprocess thoughts
				for i in range(len(allthoughts)):
					thought = allthoughts[i]
					thought = thought.lower() #make everything lower case
					thought = str(TextBlob(thought).correct()) #correct misspelled words
					thought = contractions.fix(thought)
					thought = ' '.join([num2words(word) if word.isdigit() else word for word in thought.split()])
					thought = re.sub(r'(?<=[.,])(?=[^\s])', r' ', thought)
					thought = thought.translate(str.maketrans('', '', string.punctuation))
					thought = remove_stopwords(thought)
					thought = ' '.join(thought.split())
					thought = thought.strip()
					prepthoughts.append(thought)
#				logging.info(prepthoughts)
				#tokenization and padding
				# prepare tokenizer
				with open('supp_materials/H1_train_texts.csv','rt') as f:
					reader=csv.reader(f)
					next(reader)
					train_text = []
					for row in reader:
						for item in row:
							train_text.append(item)
				f.close()
				max_words = 2000
				t = Tokenizer(num_words = max_words)
				t.fit_on_texts(train_text)
				vocab_size = len(t.word_index) + 1
				encoded_thoughts = t.texts_to_sequences(prepthoughts)
				thoughts = pad_sequences(encoded_thoughts, maxlen=25, padding='post')
				#load models
				single_models = []
				for i in range(9):
					model_name ='supp_materials/per_schema_models/schema_model_' + str(i)
					model = keras.models.load_model(model_name + '.h5')
					single_models.append(model)
				#turn each thought into array of schema scores
				all_preds = np.zeros((len(thoughts),9))
				for i in range(9):
					model = single_models[i]
					out = model.predict(thoughts)
					preds = out.argmax(axis=1)
					all_preds[:,i] = preds
				schemadist = np.amax(all_preds, axis=0)
#				logging.info("schemadist: ", schemadist)
				index_schema = np.argwhere(schemadist == np.amax(schemadist)).flatten().tolist()
				schema_list = [schemas[i] for i in index_schema]
#				logging.info("active schemas: ", schema_list)

				#spider chart
				schemas = ["Attachment","Competence","Global\n self-evaluation","Health","Power","Meta-\nCognition","Other people","Hopelessness","Other's views\n on self"]
				schema_sortkey = [0,6,8,4,1,2,3,5,7]
				schemas_reordered = [schemas[i] for i in schema_sortkey]
				c_schemadist = schemadist.astype(int).tolist()
#				logging.info("c_schemadist: ", c_schemadist)
				c_schemadist_reordered = [c_schemadist[i] for i in schema_sortkey]
				N = len(c_schemadist)
				c_schemadist_reordered += c_schemadist_reordered[:1]

				schemahistory = np.matrix(schemahistory)
				if schemahistory.size==0:
					l_schemadist_reordered = [0,0,0,0,0,0,0,0,0,0]
					markersizedist_reordered = [1,1,1,1,1,1,1,1,1,1]
				else:
					l_schemadist = np.vstack((schemahistory, schemadist))
					l_schemadist = l_schemadist.mean(0).tolist()
					l_schemadist = [item for sublist in l_schemadist for item in sublist] #remove when working with DB
					l_schemadist_reordered = [l_schemadist[i] for i in schema_sortkey]
					l_schemadist_reordered += l_schemadist_reordered[:1]
					markersizedist = schemahistory.sum(0).tolist()
					markersizedist = [item for sublist in markersizedist for item in sublist] #remove when working with DB
					markersizedist = list(map(add, markersizedist,c_schemadist))
					markersizedist = [element * 0.5 for element in markersizedist]
					markersizedist_reordered = [markersizedist[i] for i in schema_sortkey]
					markersizedist_reordered += markersizedist_reordered[:1]

				#hide dot in center for all zero labels because it was confusing for participants
				c_schemadist_reordered = [10 if x==0 else x for x in c_schemadist_reordered] #we just move the dot off of the grid that we're showing
				l_schemadist_reordered = [10 if x==0 else x for x in l_schemadist_reordered] #we just move the dot off of the grid that we're showing

				fig = plt.figure(figsize=(10,6))
				ax = plt.subplot(polar=True)

				theta = np.linspace(0, 2 * np.pi, N+1)
				lines, labels = plt.thetagrids(range(0, 360, int(360/len(schemas))), (schemas_reordered))
				area1 = 100 * np.array([1,1,1,1,1,1,1,1,1,1])**2
				ax.scatter(theta, c_schemadist_reordered, s=area1, alpha=.5)
				plt.plot(theta, l_schemadist_reordered, marker="o",color='black', alpha=0)
				area2 = 200 * np.array(markersizedist_reordered)**2
				ax.scatter(theta, l_schemadist_reordered, s=area2, alpha=.5)

				ax.tick_params(axis='x', which='major', pad=30)
				ax.set_rlabel_position(15)
				plt.xticks(theta[:-1],schemas_reordered)
				plt.yticks([0,1,2,3],color="grey",size=10)
				plt.ylim(0,3.5)
				#plt.legend(labels=('All', 'Current'), loc="lower left", bbox_to_anchor=(1,1))
				plt.tight_layout()
				filename_content = str(partID)+ "_contentfb_" + "tr" + str(tr_indx)
				imagepath_content = "fbplots/" + filename_content + '.png'
				plt.savefig(imagepath_content)
				config = {'title': filename_content} 
				cimage = client.upload_from_path(imagepath_content, config=config,anon=False)
				cimage_link = cimage['link']
				plt.close()
		return [SlotSet("posterior",c_schemadist),
		# SlotSet("schema_history", schema_history),
		# SlotSet("step_history", step_history),
		SlotSet("last_schema",schema_list),
		SlotSet("cimage_link", cimage_link),
		SlotSet("pimage_link", pimage_link)]

class ActionGiveFeedback(Action):
	def name(self) -> Text:
		return "caction_give_feedback"

	async def run(
		self,
		dispatcher: CollectingDispatcher,
		tracker: Tracker,
		domain: Dict[Text, Any],) -> List[EventType]:

		cond = tracker.get_slot("condition")
		session = tracker.get_slot("session1")
		allthoughts = tracker.get_slot("thoughts")
		schemapost = tracker.get_slot("posterior")
		cschema = tracker.get_slot("last_schema")
		tr_indx = tracker.get_slot("tr_counter")
		pimage_link = tracker.get_slot("pimage_link")
		cimage_link = tracker.get_slot("cimage_link")
		emotype = tracker.get_slot("emotype_internal")

#		logging.info(allthoughts)

		if cschema is None:
			cschema = []

		schemas = ["Attachment","Competence","Global self-evaluation","Health","Power and control","Meta-Cognition","Other people","Hopelessness","Other's views on self"]
		if len(cschema) == 1:
			resschemas = cschema[0]
		elif len(cschema) == 2:
			resschemas = cschema[0] + ' and ' + cschema[1]
		elif len(cschema) > 2:
			resschemas = cschema[0]
			i=1
			while i < len(cschema):
				if i < len(cschema)-1: 
					resschemas += ', ' + cschema[i]
				else:
					resschemas += ', and ' + cschema[i]
				i += 1
		else:
			resschemas = ""

		if session and emotype=="happy":
			utt0 = "Thank you for completing this thought record! You indicated that this scenario would not have resulted in you experiencing a negative emotion. That's great! Note, however, that thought records are the most useful when based on events that caused a negative emotion. That's because negative emotions are the ones that we are more likely to experience as bothersome. If we can discover a pattern in our thought record responses, it may be possible to work on the underlying causative beliefs. In the second session, I'll therefore ask you to focus on situations that caused a negative emotion."
		else:
			utt0 = "Thank you for completing this thought record! As you complete more thought records, you might be able to recognize recurring patterns in your responses."

		if cond == "nofb" or emotype=="happy":
			utterance = utt0
			dispatcher.utter_message(text=utterance)

		if emotype != "happy" and cond != "nofb":
			#count number of thoughts
			number_of_thoughts = len(allthoughts) - 1 #subtract 1 for automatic thought
			#utter number of characters
			utt1 = 'You completed %s downward arrow steps!' % number_of_thoughts
			#read in look-up table
			dfcomp = pandas.read_csv("supp_materials/trs_per_depth.txt", sep='\t', index_col=False)
			#get number of thought records where depth is current depth
			# tr_count = dfcomp.loc[dfcomp['levels'] == number_of_thoughts, 'trsOFdepth'].iloc[0]
			# tr_total = dfcomp.loc[:,'trsOFdepth'].sum(axis=0)
			# tr_percent = round(tr_count.astype("float")/tr_total.astype("float")*100).astype("int")
			percdata = dfcomp.loc[dfcomp['levels'] == number_of_thoughts,'cumperc'].iloc[0]
			tr_percent = str(percdata.astype("int"))+'%'
			utt2 = 'To give an indication, approximately %s of the completed thought records had the same number or fewer completed steps in a previous study.' % tr_percent
			utterance = utt0 + " " + utt1 + " " + utt2
			dispatcher.utter_message(text=utterance)
			if tr_indx == 1:
				utterance2 = "The image below shows the number of steps you have completed in this thought record. With every additional thought record you do, this plot will be expanded. On the right y-axis you can see for every step on the left y-axis the percentage of thought records in a previous study that had at least as many steps."
			else:
				utterance2 = "The image below shows the number of steps you have completed in this thought record together with those of your previous thought records. Additional thought records you complete will be added as you finish them. On the right y-axis you can see for every step on the left y-axis the percentage of thought records in a previous study that had at least as many steps."
			dispatcher.utter_message(text=utterance2)
			dispatcher.utter_message(image=pimage_link)

		if emotype != "happy" and cond == "content":
			if len(cschema) > 1:
				utterance3 = "After all you've written about your thoughts, my calculations determined that the most active core beliefs in the thought record were the %s core beliefs." % resschemas
			else:
				utterance3 = "After all you've written about your thoughts, my calculations determined that the most active core belief in the thought record was the %s core belief." % resschemas 
			dispatcher.utter_message(text=utterance3)
			if tr_indx == 1:
				utterance4 = "Below is a plot of how active (score from 0-3) I think each of the core beliefs is in this thought record. Those are the blue dots. The core beliefs are along the axes. The scores are the circles and can take the following values: 0 (core belief not detected), 1 (core belief a little bit present), 2 (core belief somewhat present), and 3 (core belief present). This plot is based only on what you have written and not on the scenario." 
			elif tr_indx == 2:
				utterance4 = "Below is the plot you've already seen after the last thought record. The blue dots again show how active (score from 0-3) I think each of the core beliefs is in the thought record you completed just now."\
				" But this time, you also see orange dot(s). The position of the orange dot(s) indicates how active I think a certain core belief was on average across all your completed thought records, "\
				"while the size of orange dots reflects the strength of this activity. So, for example, if I always think your thought records always very clearly reflect an Attachment core belief (always score 3), the " \
				"position of the orange dot will stay at 3, but its size will grow with every additional thought record that reflect an Attachment schema with score 3. Different hues of the orange circles have no meaning."\
				" Thus, the more thought records you do, the more complete the image will be."
			else:
				utterance4 = "I've again added to your core belief plot what I was able to make of your responses. The blue dots show the result of those calculations for the current thought record, the orange dots for all the thought records you completed combined, including those of the first session and the current one. The position of the orange dots reflects the average activity while their size reflects the total activity."
			dispatcher.utter_message(text=utterance4)
			dispatcher.utter_message(image=cimage_link)

		return [SlotSet("feedback_given",True),
		SlotSet("cimage_link", None),
		SlotSet("pimage_link", None)]


class ActionSaveToDB(Action):
	def name(self) -> Text:
		return "caction_savetodb"

	async def run(
		self, 
		dispatcher: CollectingDispatcher,
		tracker: Tracker,
		domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:

		partID = tracker.get_slot("participantID")
		cond = tracker.get_slot("condition")
		sess = tracker.get_slot("session1")
		tr = tracker.get_slot("tr_counter")
		sit = tracker.get_slot("asituation")
		emo = tracker.get_slot("bemotion")
		emoint = tracker.get_slot("cemointensity")
		emotype = tracker.get_slot("demotype")
		thoughts = {"thoughts":tracker.get_slot('thoughts')}
		schemadist = {"schemas": tracker.get_slot('posterior')}
		steps = tracker.get_slot('step_counter')
		behavior = tracker.get_slot("behavior")

		# for thought in thoughts:
		# 	thought.replace("'","\\'")

		# write the sql query here.
		query = """ INSERT INTO CoreData
		(`participantID`, `compDate`,`condition`, `session`, `thoughtRecord`, `situation`,
		`emotion`, `emoIntensity`, `emoType`, `thoughts`, `schemadist`, `DATsteps`, `behavior`) VALUES
		(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?) """

		args = (partID, datetime.datetime.now(), str(cond), str(sess), tr, sit, emo, emoint, emotype, json.dumps(thoughts), json.dumps(schemadist), steps, behavior)

		try:
			mydb= sqlite3.connect("ANTbot.db")
			cursor = mydb.cursor()
			logging.info("Connection established, ready to save tr data")
			cursor.execute(query, args)
			mydb.commit()
		except:
			logging.info("Error occured while connecting to database or entering data into database. Error Trace: {}".format(traceback.format_exc()))
		finally:
			if mydb:
				cursor.close()
				mydb.close()
				logging.info("SQL connection is closed")

		dbsave = True

		return [SlotSet("dbsave",dbsave), 
		SlotSet("asituation",None), 
		SlotSet("bemotion",None), 
		SlotSet("cemointensity",None),
		SlotSet("demotype",None),
		SlotSet("eautomaticthought",None),
		SlotSet("behavior",None),
		SlotSet("anxstep",None),
		SlotSet("sadangstep",None),
		SlotSet("thoughts",[]),
		SlotSet("step_counter",0),
		SlotSet("feedback_given",False)]

class ActionEndTR(Action):
	def name(self) -> Text:
		return "caction_endtr"

	async def run(
		self, 
		dispatcher: CollectingDispatcher,
		tracker: Tracker,
		domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:

		cond = tracker.get_slot("condition")
		tr_indx = tracker.get_slot("tr_counter")
		emotype = tracker.get_slot("emotype_internal")

		if cond != "content" or emotype == "happy":
			if tr_indx != 2:
				val = "ncontentfbOhappynotransition"
				#fua = "utter_ask_start_tr_nocontent"
			else:
				val = "ncontentfbOhappytransition"
				#fua = "utter_end_session1_nocontent"
		else:
			val = "contentfb"
			#fua = "utter_schemadef_q"

		return [SlotSet("emotype_internal",None), SlotSet("end_tr",val)]


class ActionExplain(Action):
	def name(self) -> Text:
		return "caction_explainschema"

	async def run(
		self, 
		dispatcher: CollectingDispatcher,
		tracker: Tracker,
		domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:

		explain = tracker.get_slot("explain")
		session = tracker.get_slot("session1")
		tr_indx = tracker.get_slot("tr_counter")

		if explain == "attachment":
			val = "An attachment core belief is activated in situations that involve people that are very dear to us." \
			"If you experience a negative emotion in such a situation, this emotion may be triggered by\n" \
			"\t 1. fears of abandonment or rejection\n" \
			"\t 2. a belief that you should never need, ask for, or accept other people’s help (compulsive self-reliance) or the opposite: a belief that you cannot achieve anything without help from the loved one (over-dependence)\n" \
			"\t 3. a belief that you are unlovable, unlikable, or unattractive\n"\
			"\t 4. a belief that you are failing in your role as partner, parent, child, sibling, or very close friend\n"\
			"\t 5. a fear of being a burden to a loved one\n"\
			"\t 6. a belief that you should be the one always providing care to your loved ones but never receiving it (compulsive caregiving)."
		elif explain == "competence":
			val = "A competence core belief is activated in situations that take place in the context of a competence domain (e.g. work, education, or other task). If you experience a negative emotion in such a situation, this emotion may be triggered by\n" \
			"\t 1. a belief that you are incompetent or failing with respect to the domain\n"\
			"\t 2. a belief that your job or competence in the domain defines your worth\n"\
			"\t 3. a belief that perfect is just good enough."
		elif explain == "global":
			val = "A global self-evaluation core belief touches on the essence of who we are and thus relates to our self-esteem. It may take the form of\n"\
			"\t 1. a belief that you are a failure/useless/inadequate in general rather than for a specific task (competence schema)\n"\
			"\t 2. a belief that you are a bad person and deserving of bad things\n"\
			"\t 3. a belief that you are deserving of blame\n"\
			"\t 4. a belief that you are stupid, horrible, selfish, terrible, or some other negative self-evaluating trait on the whole"
		elif explain == "health":
			val = "A health or medical concerns core belief pertains to our physical or mental health. It may be something along the lines of\n"\
			"\t 1. a belief/fear that you are/might become physically or mentally ill\n"\
			"\t 2. a belief/fear that you are experiencing side effects of medication\n"\
			"\t 3. a fear of disability or physical pain\n"\
			"\t 4. a belief/fear related to your weight\n"\
			"\t 5. a belief/fear related to your sanity"
		elif explain == "power":
			val = "A control or power core belief that triggers a negative emotion commonly reflects themes of powerlessness, e.g.:\n"\
			"\t 1. a belief that you are not in control or feeling powerless, weak, or overwhelmed\n"\
			"\t 2. a sense that you are acting out of duty or lacking a choice\n"\
			"\t 3. a feeling of being trapped"
		elif explain == "meta":
			val = "A meta-cognition core belief reflects self-insight or the observation of one’s own thought processes (e.g., \'I have negative thinking,\'' or \'I have a need to be needed\')."
		elif explain == "others":
			val = "Sometimes general beliefs about the nature of other people or our own social nature can make us feel in situations that involve others. Such core beliefs may be:\n"\
			"\t 1. a belief that other people in general are bad/malicious/hurtful/disliking\n"\
			"\t 2. a belief that you are just not a people person or do not get on with people."
		elif explain == "hope":
			val = "A hopelessness core belief indicates that the future looks bleak to you. Such core beliefs commonly have themes of hopelessness (e.g. \'there’s no point to life\', \'life has no meaning\', \'what’s the point\', \'there’s nothing I can do\', \'life is not worth living\', \'I won\'t be able to cope\', \'no faith\', \'no future\'), inevitability, or fear (e.g. \'I’m always a target\', \'I’m not safe alone\'), but may also just indicate that you are overwhelmed by raw emotion (e.g. panic, anxiety, sadness, depression)."
		elif explain == "self":
			val = "Refers to the importance of what others think about the self (e.g. \'I care too much what others think of me\', \'people think I am …\'')"
		elif explain == "none":
			val = "alright"
		else:
			val = "I didn't get that, let's try again."

		if session and tr_indx == 1:
			if explain != "none":
				transval = "session1tr1"
			else:
				transval = "session1tr1none"
		elif session and tr_indx == 2:
			if explain != "none":
				transval = "session1tr2"
			else:
				transval = "session1tr2none"
		else:
			if explain != "none":
				transval = "session2"
			else:
				transval = "session2none"

		dispatcher.utter_message(text=val)

		return [SlotSet("explain", None), SlotSet("transition",transval)]

class ActionSaveSessionEnd(Action):
	def name(self) -> Text:
		return "caction_savesessionend"

	async def run(
		self, 
		dispatcher: CollectingDispatcher,
		tracker: Tracker,
		domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:

		partID = tracker.get_slot("participantID")
		tr = tracker.get_slot("tr_counter")
		dbsave = tracker.get_slot("dbsave")

		try:
			mydb= sqlite3.connect("ANTbot.db")
			cursor = mydb.cursor()
			logging.info("Connection established, ready to save session end")
			args = [partID]
			if dbsave:
				if tr == 1 or tr == 0:
					query = 'update Demographics set s1comp = 0, s2comp = 0 where participantID = ?'
				elif tr == 2:
					query = 'update Demographics set s1comp = 1, s1compDate = ? where participantID = ?'
					args = [datetime.datetime.now(),partID]
				else:
					query = 'update Demographics set s2comp = 1, s2compDate = ? where participantID = ?'
					args = [datetime.datetime.now(),partID]
			else:
				if tr == 3:
					query = 'update Demographics set s2comp = 0 where participantID = ?'
				elif tr > 3:
					query = 'update Demographics set s2comp = 1, s2compDate = ? where participantID = ?'
					args = [datetime.datetime.now(),partID]
				else:
					query = 'update Demographics set s1comp = 0, s2comp = 0 where participantID = ?'
			cursor.execute(query, args)
			mydb.commit()
			dbsave = False
		except:
			logging.info("Error occured while connecting to database or entering data into database. Error Trace: {}".format(traceback.format_exc()))
		finally:
			if mydb:
				cursor.close()
				mydb.close()
				logging.info("MySQL connection is closed")


		return [SlotSet("dbsave",dbsave)]

