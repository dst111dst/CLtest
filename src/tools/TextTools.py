import string
import collections
from nltk.corpus import stopwords
from nltk.tokenize import wordpunct_tokenize

def Token(words: str) -> list:
    stops = set(stopwords.words('english') + list(string.punctuation))
    words = [word for word in wordpunct_tokenize(words.lower()) if word not in stops]
    return words

def delDuplicate(words: list) -> list:
    """ Remove duplicated words, first remove front ones (for query only). """
    words_unique = []
    for word in words[::-1]:
        if word not in words_unique:
            words_unique.append(word)
    words_unique.reverse()
    return words_unique
    

def Filter(document: list, min_num: int) -> list:
    """ Filter words in documents less than min_num. """
    cnt = collections.Counter()
    for sentence in document:
        cnt.update(sentence)

    s = set(word for word in cnt if cnt[word] < min_num)
    document = [[word for word in sentence if word not in s] for sentence in document]
    return document

def CommaFilter(info:list) -> list:
    temp = []
    for x in info:
        if (isinstance(x,list)):
            for item in x:
                temp = [item for item in x]
        else:
            temp.append(x)
    res = []
    for word in temp:
        word = str(word)
        word.replace('"','')
        word.replace('"','')
        word.replace("'",'')
        word.replace("[",'')
        word.replace("]",'')
        res.append(word)
    return res


if __name__ =='__main__':
    # nltk.data.path.append(r"C:\Program Files\Anaconda3\Lib\nltk_data")
    w = ['I','have','a dream']
    l = CommaFilter(w)
    print(l)