#================================================================================
# These python functions are used by the Main ipython notebook
#================================================================================
import re
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer
import numpy as np
import collections


try:
	stopwords.words('english')
except LookupError:
	import nltk
	nltk.download('stopwords')

# This function will read in all of the fake news titles and create a large array
def aggregate_data(df, col="title",size=20):
	
	# The Regex Filter, to remove numbers from string array
	regex = re.compile(r'[a-zA-Z]+')
	
	array = []
	
	#title_text = str(df.iloc[[k]][col])[1:]
	#print title_text.split()[:-4]
	# max size: df.shape[0]
	for k in range(size):
		title_text = str(df.iloc[[k]][col])[1:]
		title_array = title_text.split()[:-4]
		
		# This is used to check validity of the extracted data
		assert title_text.split()[-4:] == ['Name:', 'title,', 'dtype:', 'object']
		
		# Now we filter out numbers from the extracted data
		title_array = filter(regex.match, title_array)
		

		array.extend(title_array)
	
	return array



# This function will take an array of words, and create a dictionary. The data will be processed as follows:
# 1. make all words into lowercase
# 2. remove stop words
def create_dictionary(word_array):
	
	new_array = []
	
	tokenizer = RegexpTokenizer(r'\w+')
	#tokenizer.tokenize('Eighty-seven miles to go, yet.  Onward!')
	
	# Remove the stop words 
	filtered_words = [word.lower() for word in word_array if word not in stopwords.words('english')]
	
	
	
	# Remove the punctuations
	#tokenizer.tokenize('Eighty-seven miles to go, yet.  Onward!')
	for word in filtered_words:
		
		# Remove any occurence of unicode characters
		word_f = word.replace("\xe2","")
		word_f = word_f.replace("\xc3","")
		word_f = word_f.replace("\xc2","")
		
		word_array = tokenizer.tokenize(word_f)
		
		# Remove any occurence of unicode '\xe2'
		
		new_array.extend(word_array)
	return new_array



# Using the dictionary, turn a corpus of text into a vector based on the dictionary
def create_feature_vector(text,dictionary):
	
	new_array = []
	vector = np.zeros(len(dictionary))
	
	#new_array.extend(word_array)
	# The Regex Filter, to remove numbers from string array
	regex = re.compile(r'[a-zA-Z]+')
	
	# Now we remove the numbers from a corpus
	text_array = text.split()
	
	#print 'text-array before Regex:', text_array
	
	text_array = filter(regex.match, text_array)
	
	#print 'text-array after Regex:', text_array
	
	# Now we make a filter for the tokenizer
	tokenizer = RegexpTokenizer(r'\w+')
	
	# Remove the stop words from the corpus
	filtered_words = [word.lower() for word in text_array if word not in stopwords.words('english')]
	
	#print 'filtered words', filtered_words
	
	for word in filtered_words:
		word_f = word.replace("\xe2","")
		word_f = word_f.replace("\xc3","")
		word_f = word_f.replace("\xc2","")
		word_array = tokenizer.tokenize(word_f)
		new_array.extend(word_array)
	
	# Count the number of occurences of different words in the new vector array
	new_counter = collections.Counter(new_array)
	
	# Turn our dictionary into something iterable
	new_counter = new_counter.most_common(len(new_counter))
	
	#print 'new_counter', new_counter
	
	for k in range(len(dictionary)):
		
		key_word = dictionary[k][0]
		
		#print 'key_word', key_word
		
		# Is the key word in the new counter array?
		for i in range(len(new_counter)):
			
			#print 'new item', new_counter[i][0]
			
			# Loop through all the words in the new corpus 
			if new_counter[i][0]== key_word:
				vector[k]= new_counter[i][1] # place the frequency into the vector

	return vector

#=======================================================================
# Create a feature Matrix based on a dataframe and a dictionary
# This function will read in a dataframe
# A[i,j], i=title entry, j=index of the vector
# A = [ \vec{v}_i for i in data_frame ]
#=======================================================================
def create_featureMatrix(df,dictionary,col='title',size=20):
	
	# Initialize the feature matrix
	features_matrix = np.zeros((size,len(dictionary)))
	
    # Generate a vector based on the colum data
	for k in range(size):
		text = str(df.iloc[[k]][col])[1:] # Extract text from column
		vec_k = create_feature_vector(text,dictionary) 
		features_matrix[k,:] = np.asarray(vec_k)
	
	return features_matrix

# This function will take in a data frame and extract the label
def create_label_array(df,col='label',size=20):
	
	# Initialize the label vector
	label_array = np.zeros(size)
	
	for k in range(size):
		label_k = int(df.iloc[k]['label'])
		label_array[k]= label_k
	
	return label_array

# Make predictions using a corpus of text and a given model
def generate_predictions(text,dictionary, model):
	
	# convert the text into a vector
	vector = create_feature_vector(text,dictionary).reshape(1,-1) 
	
	result = model.predict(vector)
	
	return result
	
# Make predictions using a corpus of text and a given model, return the probability of the classes
def generate_predictions_prob(text,dictionary, model):
	
	# convert the text into a vector
	vector = create_feature_vector(text,dictionary).reshape(1,-1)
	
	result = model.predict_proba(vector)
	
	return result
	

# This function takes as input an array of models, and will return the 
# prediction of all models as a single vector
def generate_prediction_ensemble(text,dictionary,model_array):
	
	prediction_vector = []
	
	# convert the text into a vector
	vector = create_feature_vector(text,dictionary).reshape(1,-1) 
	
	# Loop through all models and make a prediction
	# All results will be appended together
	for m in model_array:
		 model_pred = m.predict(vector)
		 prediction_vector.extend(model_pred)
	
	return prediction_vector

#=======================================================================
# Given an Ensemble of Models {M0,M1,...,Mn}, Compute:
# 1. Accuracy of Binary Classification
# 2. False Positive and False Negative Probabilities
#
# Use 1 and 2 as Priors for each model, then compute two probabilities:
# A. The probability that given a sequence of test results, the final result is: TRUE
# B. The probability that given a sequence of test results, the final result is: FALSE
# Return a vector with the [Real News Prob,Fake New Prob]
#
# Input: Text to be Classified, dictionary that is being used, and an array of models
#=======================================================================
def Bayesian_Sequence_Prediction(text,dictionary,model_array,cm_array):
	
	# Prior probability of a story being fake or real: 50-50
	prior_truth = 0.5
	prior_false = 0.5
	
	# Generate the Prediction of all models in the Ensemble
	prediction_array = generate_prediction_ensemble(text,dictionary,model_array)
	
	# Initialize the Probabilities
	prob_fake_cond_results = 0.0#prior_truth
	prob_true_cond_results = 0.0#prior_false
	Normalization =1.0
	
	# Compute the conditional probabilities for the results of all models
	for k in range(len(prediction_array)):
		
		# This means the story is True
		if prediction_array[k]==0:
			cond_prob_result_given_false_k = cm_array[k][0,1] # Wrong Classifications
			cond_prob_result_given_true_k  = cm_array[k][0,0] # Correct Classifications
		# The story is false
		elif prediction_array[k]==1:
			cond_prob_result_given_false_k = cm_array[k][1,1] # Correct Classifications
			cond_prob_result_given_true_k  = cm_array[k][1,0] # Wrong Classifications
		
		#print cond_prob_result_given_true_k,cond_prob_result_given_false_k
			
		# The Total Bayesian Product
		prob_fake_cond_results = prob_fake_cond_results+cond_prob_result_given_false_k
		prob_true_cond_results = prob_true_cond_results+cond_prob_result_given_true_k	
	
	Normalization = prob_fake_cond_results+prob_true_cond_results
	
	prob_fake_cond_results = prob_fake_cond_results/Normalization
	prob_true_cond_results = prob_true_cond_results/Normalization
	
	
	
	return [prob_true_cond_results,prob_fake_cond_results]




	


