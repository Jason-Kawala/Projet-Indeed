#import modules
import os.path
from gensim import corpora
from gensim.models import LsiModel
from nltk.tokenize import RegexpTokenizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from gensim import matutils, models
import scipy.sparse
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from gensim.models.coherencemodel import CoherenceModel
#import matplotlib.pyplot as plt
import pandas as pd
#import numpy as np
#import datetime
import warnings
from IPython.display import display
pd.set_option('display.max_rows', 1000)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)
warnings.simplefilter(action='ignore')

# import libraries for NLP
import spacy
from spacy import displacy
nlp = spacy.load('fr_core_news_md')
import nltk
#from nltk.stem.snowball import SnowballStemmer
from nltk.stem.snowball import FrenchStemmer
from spacy.lang.fr.stop_words import STOP_WORDS as fr_stop
stemmer = FrenchStemmer()
tokenizer = nltk.RegexpTokenizer(r'\w+')
stop_fr  = nltk.corpus.stopwords.words('french')
stop_uk  = nltk.corpus.stopwords.words('english')
stop_spacy_fr=list(fr_stop)

class TopicExtract:
    def __init__(self,location):
        self.uniqueLocation=list(location.unique())
    # put all in lower case
        self.uniqueLocation=[x.lower() for x in self.uniqueLocation]

    def parse_text(self,x):
        #df['Location']=df['Location'].str.lower()
        text=str(x)
        text = text.replace(r'[,\.!?0-9]',' ') 
    # tokenize the text
        text = tokenizer.tokenize(text.lower())
    # delete english and frensh stop words
        text = [word for word in text if not word in stop_fr]
        text = [word for word in text if not word in stop_uk]
        text = [word for word in text if not word in stop_spacy_fr]
    # delete not wanted chars/words
        text = [word for word in text if not word in ['h','hf', 'f', 'euse', 'se', 'e']]
        text = [word for word in text if not any(word in s for s in self.uniqueLocation)]
    # delete not chars
        text=[word for word in text if word.isalpha()]
    # keep words with length more than 2 chars
        text=[word for word in text if len(word)>2]
        #text = [stemmer.stem(word) for word in text]
        return text
	
    def concatExtractedWords(self,title,summary):
        concatWords=self.parse_text(title)+self.parse_text(summary)
    #concatWords.append(parse_text(title,location))
    #concatWords.append(parse_text(summary,location))
    #print(concatWords)
        return concatWords
	
    def generateBOW(self,df):
        df['BagOfWords'] = df.apply(lambda x: self.concatExtractedWords(x.Title, x.Description), axis=1)
        return df

    def prepare_corpus(self,doc,gram=(1,2),option='c'):

        if option =='c':
    # Create a new document-term matrix using only nouns and adjectives, also remove common words with max_df
            cvna = CountVectorizer(tokenizer=self.parse_text,ngram_range=gram,stop_words=stop_fr,strip_accents='ascii', max_df=.8)
            data_cvna = cvna.fit_transform(doc)
            data_dtmna = pd.DataFrame(data_cvna.toarray(), columns=cvna.get_feature_names())
		# Create the gensim corpus (term_document matrix)
            doc_term_matrix = matutils.Sparse2Corpus(scipy.sparse.csr_matrix(data_dtmna.transpose()))
        # Create the vocabulary dictionary
            dictionary = dict((v, k) for k, v in cvna.vocabulary_.items())
        elif option =='tf':
            tfna= TfidfVectorizer(tokenizer=self.parse_text,ngram_range=gram,stop_words=stop_fr,strip_accents='ascii', max_df=.8)
            data_tfna = tfna.fit_transform(doc)
            data_dtmna = pd.DataFrame(data_tfna.toarray(), columns=tfna.get_feature_names())
        # Create the gensim corpus (term_document matrix)
            doc_term_matrix = matutils.Sparse2Corpus(scipy.sparse.csr_matrix(data_dtmna.transpose()))
        # Create the vocabulary dictionary
            dictionary = dict((v, k) for k, v in tfna.vocabulary_.items())
    # generate LDA model
        return dictionary,doc_term_matrix

	
    def getLSATopics(self,doc,number_of_topics,chunk=2000,gram=(1,2),option='c'):
	    dictionary,doc_term_matrix=self.prepare_corpus(doc,gram,option)
	# generate LSA model
	    lsa = LsiModel(doc_term_matrix, num_topics=number_of_topics, id2word = dictionary,chunksize=chunk)  # train model
	    display(lsa.print_topics())
    # Let's take a look at which topics each transcript contains
	    corpus_transformed = lsa[doc_term_matrix]
    # transform the result into numpy array to get the score for each title 
	    all_topics_csr = matutils.corpus2csc(corpus_transformed)
	    all_topics_numpy = all_topics_csr.T.toarray()
    #Lsa_Topic=pd.DataFrame(all_topics_numpy)
	    Lsa_Topic=pd.DataFrame(all_topics_numpy,doc)
	    display(Lsa_Topic.head(5))
	    print('shape ',Lsa_Topic.shape)
	    return Lsa_Topic

	
    def getLDaTopics(self,doc,number_of_topics,passe=20,iters=100,chunk=2000,gram=(1,2),option='c'):
	    dictionary,doc_term_matrix=self.prepare_corpus(doc,gram,option)
		# generate LDA model
	    lda = models.LdaModel(corpus=doc_term_matrix, id2word=dictionary,num_topics=number_of_topics,iterations=iters,passes=passe,chunksize=chunk,random_state=1)# train model
		#print(ldamodel.print_topics(),'\n')
	    display(lda.print_topics())
	    corpus_transformed = lda[doc_term_matrix]
	    all_topics_csr = matutils.corpus2csc(corpus_transformed)
	    all_topics_numpy = all_topics_csr.T.toarray()
    #Lda_Topic=pd.DataFrame(all_topics_numpy)
	    Lda_Topic=pd.DataFrame(all_topics_numpy,doc)
	    display(Lda_Topic.head(5))
	    print('shape ',Lda_Topic.shape)
	    return Lda_Topic
	

	

	


	
	
	
