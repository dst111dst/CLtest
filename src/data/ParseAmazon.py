import gzip
import json
import random
import itertools
import pandas as pd
from tqdm import tqdm
from typing import Tuple,Dict,List
from src.tools.TextTools import *

def ParseReviews(name:str) -> pd.DataFrame:
    """
    Parse the raw Reviews data to pandas DataFrame.
    :param name is the name of the datasets, e.g. Cell_Phones_and_Accessories
    :return:
        pd.Dataframe, dict-like: info-id,info
    """
    idx, TempDf = 0, dict()
    # path = 'data/raw/reviews_'+name+'.json.gz'
    path = name
    with gzip.open(path,'rb') as gf:
        progress = tqdm(gf,desc = 'parsing reviews to a dataframe',unit_scale=True,total=len(gzip.open(path, 'rb').readlines()))
        for line in progress:
            TempDf[idx]=json.loads(line)
            idx += 1
    return pd.DataFrame.from_dict(TempDf, orient='index')


def ParseMeta(name: str) -> Tuple[Dict[str, str], Dict[str, str], Dict[str,List[str]]]:
    """
    Extract useful information (i.e., categories, related) from meta file.
    :param name is the name of the datasets, e.g. Cell_Phones_and_Accessories
    :return:
        categories: dict->itemID: category, this is for query.
        also_viewed: dict->itemID: also_viewed items
        attributes: other attributes?
    """
    categories, also_viewed, attributes = {}, {}, {}
    meta_path = name
    with gzip.open(meta_path, 'rb') as gf:
        progress = tqdm(gf, desc='parsing meta',
                        unit_scale=True,
                        total=len(gzip.open(meta_path, 'rb').readlines()))
        for line in progress:
            line = eval(line)
            if 'asin' not in line:
                pass
            asin = line['asin']
            if 'category' in line:
                categories[asin] = line['category']
                attributes[asin] = line['brand'] if 'brand' in line else None
            elif 'categories' in line:
                # ONLY SAVE ONE CATEGORY..?
                # line['categories'] = CommaFilter(line['categories'])
                categories[asin] = random.choice(line['categories'])
                if len(line['categories']) > 1:
                    attributes[asin] = line['categories'][1:]
                else:
                    attributes[asin] = [c for c in line['categories'][0][1:]]
                if 'brand' in line:
                    attributes[asin].append(line['brand'])
            else:
                raise Exception('categories tag not in metadata')
            related = line['related'] if 'related' in line else None
            titles = line['title'] if 'title' in line else None
            also_viewed[asin] = []  # Next fill the also_related dictionary
            relations = ['also_viewed', 'buy_after_viewing']  # consider 2 relations.
            if related:
                also_viewed[asin] = [related[r] for r in relations if r in related]
                also_viewed[asin] = itertools.chain.from_iterable(also_viewed[asin])
            if titles:
                attributes[asin].append(titles)
        return categories, also_viewed, attributes


def ParseWords(reviewDf: pd.DataFrame, min_words: int, categories: Dict[str, str], attributes:Dict[str,List[str]],defaultQuery:List[str]) -> Tuple[
    pd.DataFrame, Dict[str, int]]:
    """
    Parse words for reviews and queries. For better comparing with other baselines,
    we don't change the methods of extracting queries.
    :param reviewDf: reviews dataframe from function 'parseReviews', each row is identified
    by item-id and contains reviews.
    :param min_words: the minimum words of a review.
    :param categories: the categories of an item
    :param defaultQuery: the default query we set for items.

    :return:
        reviews -> after processed
        words - > in queries and reviews.
    """
    queries,reviews,attrs = [], [], []
    word_set = set()
    progress = tqdm(range(len(reviewDf)),
                    desc='parsing review and query words',
                    total=len(reviewDf), unit_scale=True)
    for i in progress:
        asin = reviewDf['asin'][i]
        r = reviewDf['reviewText'][i]
        category = categories[asin] if categories.get(asin) else defaultQuery
        category = ' '.join(map(str,category))
        a1 = ' '.join(map(str,attributes[asin]))
        query = delDuplicate(Token(category))
        a = delDuplicate(Token(a1))
        for word in query:
            word_set.add(word)
        for item in a:
            word_set.add(item)
        # process reviews
        review = Token(r)
        queries.append(query)
        reviews.append(review)
        attrs.append(a)

    reviewDf['query'] = queries  # write query result to dataframe
    reviewDf['attributes'] = attrs

    # filtering words counts less than min_num
    reviews = Filter(reviews, min_words)
    for review in reviews:
        for word in review:
            word_set.add(word)
    reviewDf['reviewText'] = reviews
    word_dict = dict(zip(word_set, range(len(word_set))))  # start from 0
    reviewDf['queryWords'] = [[word_dict[word] for word in query] for query in queries]
    reviewDf['attrWords'] = [[word_dict[word] for word in attribute ] for attribute in attrs ]
    print(reviewDf.head())
    return reviewDf, word_dict

def ReviewMap(input_df:pd.DataFrame) -> pd.DataFrame:
    """ 
    Reindex the reviewID from 0 to total length to build the id2map.
    :param input_df: the input dataframe of reviews.
    :return:
        a map of dataframe, each row's number corresponds to a review.
    """
    review = input_df['reviewerID'].unique()
    review_map = dict(zip(review, range(len(review))))

    userIDs = [review_map[input_df['reviewerID'][i]] for i in range(len(input_df))]
    input_df['userID'] = userIDs
    return input_df

def SplitData(input_df:pd.DataFrame,ratio:float) -> pd.DataFrame:
    """
    Splitting data into training and testing based on the ratio we set.
    :param input_df: the data we input.
    :param ratio: the split ratio of training and testig set.
    :return:
        a dataframe with an extra axis of testing/traning information.
    """
    split_indicator = []
    df = input_df.sort_values(by=['userID', 'unixReviewTime'])
    user_length = df.groupby('userID').size().tolist()
    progress = tqdm(range(len(user_length)), desc='splitting data', total=len(user_length), unit_scale=True)
    for index in progress:
        length = user_length[index]
        if length == 1:
            tag = ['Train']
        else:
            tag = ['Train' for _ in range(int(length * 0.7))]
            tag_test = ['Test' for _ in range(length - int(length * 0.7))]
            tag = ['Train' for _ in range(length - 1)]
            tag_test = ['Test']
            tag.extend(tag_test)
        # np.random.shuffle(tag)
        split_indicator.extend(tag)
    df['filter'] = split_indicator
    return df

def DelDupRev( input_df:pd.DataFrame, word_dict:Dict[str,int] ) -> pd.DataFrame:
    review_text, review_words, review_train_set = [], [], set()
    df = input_df.reset_index(drop=True)
    df_test = df[df['filter'] == 'Test'].reset_index(drop=True)
    # review in test
    review_test = set(repr(df_test['reviewText'][i]) for i in range(len(df_test)))
    progress = tqdm(range(len(df)), desc='removing reviews', total=len(df), unit_scale=True)
    for i in progress:
        r = repr(df['reviewText'][i])
        if r not in review_train_set and r not in review_test:
            review_train_set.add(r)
            review_text.append(eval(r))
            review_words.append([word_dict[w] for w in eval(r)])
        else:
            review_text.append([])
            review_words.append([])
    df['reviewText'] = review_text
    df['reviewWords'] = review_words
    return df


