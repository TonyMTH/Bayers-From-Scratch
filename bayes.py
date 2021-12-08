#import libraries
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import nltk
nltk.download('stopwords')
nltk.download('punkt')

from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from collections import Counter

class NB:

    train_X = None
    y_train = None

    def fit(self,train_X,y_train):
        self.train_X = train_X
        self.y_train = y_train
        return self
    
    def pre_process(self,sentence):
        porter = PorterStemmer()
        words = nltk.word_tokenize(sentence)
        stop = stopwords.words('english')
        words=[porter.stem(word.lower()) for word in words if word.isalpha() and word not in stop]
        return words

    def get_ham_spam(self,train_X,y_train):
        ham_spam = {0:[],1:[]}
        for id,l in enumerate(y_train):
            ham_spam[l].extend( self.pre_process(train_X[id]) )
        return ham_spam[0],ham_spam[1]

    def get_ham_spam_unprocessed(self,train_X,y_train):
        ham_spam = {0:'',1:''}
        for id,l in enumerate(y_train):
            ham_spam[l] += train_X[id]
        return ham_spam[0],ham_spam[1]

    def get_words_frequency(self,word_list):
        return {k:v+1 for k,v in Counter(word_list).items()}

    def probWord_given_Ham_or_Spam(self,word,word_freq,word_collection):
        collection_freq = self.get_words_frequency(word_collection)
        try:
            word_freq_in_collection = collection_freq[word]
        except:
            word_freq_in_collection = 1
        total_word_collection =  sum(collection_freq.values())
        return (word_freq_in_collection/total_word_collection)**word_freq

    def prob_Ham_or_Spam(self,ham_freq,spam_freq):
        total_ham_words_count =  sum(ham_freq.values())
        total_spam_words_count =  sum(spam_freq.values())
        total_words_count = total_ham_words_count + total_spam_words_count
        return total_ham_words_count/total_words_count, total_spam_words_count/total_words_count

    def probHamOrSpam_given_sentence(self,sentence):
        word_list = self.pre_process(sentence)
        words_frequency = self.get_words_frequency(word_list)
        return words_frequency

    def __predict_one(self,sentence,ham,spam,prob_Ham,prob_Spam):
        word_count = self.probHamOrSpam_given_sentence(sentence)

        probWords_given_Ham = 1
        probWords_given_Spam = 1
        for k,v in word_count.items():
            probWords_given_Ham *= self.probWord_given_Ham_or_Spam(k,v,ham)
            probWords_given_Spam *= self.probWord_given_Ham_or_Spam(k,v,spam)

        probWords_given_Ham = prob_Ham*probWords_given_Ham
        probWords_given_Spam = prob_Spam*probWords_given_Spam

        return probWords_given_Ham, probWords_given_Spam, int(probWords_given_Ham<probWords_given_Spam)

    def predict(self,x_test):
        #common computations
        ham,spam = self.get_ham_spam(self.train_X,self.y_train)
        ham_freq = self.get_words_frequency(ham)
        spam_freq = self.get_words_frequency(spam)
        prob_Ham, prob_Spam = self.prob_Ham_or_Spam(ham_freq,spam_freq)

        y_pred = []
        for x in x_test:
            yp = self.__predict_one(x,ham,spam,prob_Ham,prob_Spam)
            y_pred.append(yp[-1])
        return y_pred

    def evaluate(self,y, y_p):
        return sum(y == y_p) / len(y)

    def text_by_image(self,id='h',count=20,text=None):
        ham,spam = self.get_ham_spam_unprocessed(self.train_X,self.y_train)
        if id == 'h':
            text = ham
        elif id == 's':
            text = spam
        else:
            text = text
        wordcloud = WordCloud(max_font_size=50, max_words=count, background_color="white").generate(text)
        _, ax = plt.subplots(figsize=(12,10))
        ax.imshow(wordcloud, interpolation='bilinear')
        ax.axis("off");

