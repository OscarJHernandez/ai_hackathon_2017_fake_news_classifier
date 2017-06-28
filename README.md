# ai_hackathon_2017_fake_news_classifier
This repository contains the data files and  including the weights of the pre-trained models that were used to create a model that could detect fake news 



The steps of the model are as follows:
## Step 1: Dictionary Creation
* The user specifies the number of headlines to train the model on (N).
* The user must also specify the size of the dictionary to create (D)
* A csv file containing news headlines, both fake and real, are loaded in.
* N headlines are read in, and each headline has its punctuation removed, stop words removed, unicode characters removed.
* All of words are stored in an array and a dictionary is then created which counts the frequency of all the words that it read in.
* The dictionary will order the words according to the frequency of their occurence.

## Step 2: Feature Matrix Calculation
* Now that we have a dictionary, when we take a headline, we can map it into a vector that counts the words in the dictionary
Example: Suppose the dictionary is: {the, trump, apple,ate, orange}, then the text = "I ate an apple apple and then I ate an Orange" would be mapped to the vector, v = [0,1,2,1].
* Now using the training set that we used to create a dictionary, each headline is read in, and mapped into a vector. Each new headline which corresponds to a new vector is added to the rows of a matrix. The dimensions of this matrix M are: [M] = [N x D]

## Step 3: Training
* Using the generated feature matrix, we must then create a vector of dimension [D], which contains a 0 or 1, indicating the label of the headline. 0= True, 1 = False.
* The feature matrix with the feature vector is loaded into a scikit learn classifier that trains the model. ( model.fit(feature_matrix, feature_vector))
* The model is then used to make predictions on a test data set, by generating the feature matrix for the test data, and then predicting the labels.
* I then extract the confusion matrix which tells us the correct identifications in the test training set, along with false positive, and false negative rates.
* The final prediction uses all the models weighed according to their accuracy and false positive rates to give a final value and produce a number that indicates the confidence of the classification.





