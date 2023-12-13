import re
import string
import nltk
import contractions
from nltk.corpus import stopwords
from gensim.parsing.preprocessing import remove_stopwords

nltk.download('stopwords')

def clean_data(data):
    """Function to apply regex operations to make the data more useable"""

    data = data.lower() # lowercase 
    data = re.sub('\[.*?\]', '', data) # remove text in square brackets
    data = re.sub('[%s]' % re.escape(string.punctuation), '', data) # remove punctuation
    data = re.sub('\w*\d\w*', '', data) # remove words containing numbers
    data = re.sub('[‘’“”…]', '', data) # remove quotations
    data = re.sub('\n', '', data)
    data = remove_stopwords(data)
    data = contractions.fix(data)
    return data