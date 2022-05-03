import random
import pickle
from numpy.linalg import norm
import scipy.sparse as sp
from typing import Dict,List
from tqdm import tqdm
import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
from src.tools.Params import building_progress
from src.tools.Sample import *

# where to put attributes???
def Padding(batch,value):
    items,words,queries, attrs = zip(*batch)
    items = torch.stack(attrs)
    queries = torch.stack(queries)
    words = pad_sequence(words, batch_first=True, padding_value=value)
    return items, words, queries

def train_similarity(item_dict: dict, most_similar: int, save_path: str = './similarity.pkl') :
    l = len(item_dict)
    res = dict()
    if most_similar > l :
        most_similar = l
    max_num = np.float(1e5)
    for idx in range(l):
        res.setdefault(idx,[])
        a = item_dict[idx]
        x = np.zeros((l,1))
        count = 0
        for v in item_dict.values():
            try:
                X =  np.inner(a, v)/(norm(v)*norm(a))
                x[count] =X
                count += 1
            except Exception as e:
                x[count] = max_num
                count += 1
                continue
        temp = (x.argsort(axis=0) )# the most similar items. BUT IT MIGHT not be resonable to selecy the ons with highest similarity.
        res[idx]  = ((temp[1:1+most_similar]).flatten()).tolist()
        print(res[idx])
    print("saving data to ", save_path)
    with open(save_path, 'wb') as write_file:
        pickle.dump(res, write_file)

class AmazonDataset(Dataset):
    def __init__(self, input_df:pd.DataFrame,
                 users:list, item_map: Dict[str,int],
                 TotalWords:int, QueryMax:int,
                 SentenceMax:int, AttrMax:int,
                 Mode:str, isDebug:bool, userBuy:Dict = None):
        self.Mode = Mode
        self.userSeq = users
        self.itemMap = item_map
        self.wordNum = TotalWords
        self.userBuy = userBuy
        self.userNum = len(users)
        self.itemNum = len(item_map)
        self.dataSave = []

        def ExtractSentence(sentence:list, max_len:int):
            sentence = sentence[: max_len] if len(sentence) > max_len else \
                sentence + [TotalWords] * (max_len - len(sentence))
            return torch.tensor(sentence, dtype=torch.long)

        if Mode == 'train':
            progress = building_progress(input_df, isDebug, desc='iter train')
            self.userBuy= dict()
            # load u-i interactions as a dok matrix - for negative sampling
            self.u2iMatrix = sp.dok_matrix((self.userNum, self.itemNum), dtype=np.float32)
            # adjacency matrix - for GCN estimation
            self.adjMatrix = sp.dok_matrix((self.userNum + self.itemNum,
                                             self.userNum + self.itemNum),
                                            dtype=np.float32)
            item2attribute = dict.fromkeys(range(self.itemNum)) # the int
            # u2i Matrix is a graph for user and items
            # so why we not construct a graph to record the items? -> feature levels
            for _, entry in progress:
                user = entry['userID']
                item = self.itemMap[entry['asin']]
                words = eval(entry['reviewWords'])
                attri = eval(entry['attrWords'])
                self.u2iMatrix[user, item] = 1.0
                user += self.itemNum    # entity - [item; user]
                if user not in self.userBuy:
                    self.userBuy[user] = []
                if item in self.userBuy[user]:
                    self.userBuy[user].remove(item)
                self.userBuy[user].append(item)  #Items are appended chronologically
                self.adjMatrix[user, item] = 1.0
                self.adjMatrix[item, user] = 1.0

                attr_vec = ExtractSentence(attri,AttrMax)
                temp = attr_vec.cpu().numpy()
                self.i2adim = temp.shape[0]
                item2attribute[item] = temp.reshape(1,self.i2adim)
                # if item2attribute[item] is None:
                #     print(item,attr_vec)
                # print(item2attribute[item].shape)
                words = ExtractSentence(words, SentenceMax)
                query = ExtractSentence(eval(entry['queryWords']), QueryMax)
                # attribute = ExtractSentence(eval(entry['attrWords']),AttrMax)
                self.dataSave.append({
                    'user': user,
                    'item': item,
                    'word': words,
                    'query': query,
                    'attribute' :attr_vec
                })
            # self.item2attribute = pd.DataFrame.from_dict(item2attribute, orient='index')
            self.item2attribute = item2attribute
            self.all_attributes = np.zeros((self.itemNum, self.i2adim))
            i = 0
            for v in self.item2attribute.values():
                self.all_attributes[i] = v
                i += 1
            # raise ValueError

        elif self.Mode == 'test':
            progress = building_progress(input_df, isDebug, desc='iter test')
            for _, entry in progress:
                user = entry['userID']
                item = item_map[entry['asin']]
                user += self.itemNum
                query = ExtractSentence(eval(entry['queryWords']), QueryMax)
                attribute = ExtractSentence(eval(entry['attrWords']), AttrMax)
                self.dataSave.append({
                    'user': user,
                    'item': item,
                    'query': query,
                    'attribute':attribute
                })

    def __len__(self):
        return len(self.dataSave)

    def __getitem__(self, index):
        entry = self.dataSave[index]
        if self.Mode == 'train':
            return entry['user'], entry['item'], entry['query'],\
                   entry['word'], entry['item_neg']
        else:
            return entry['user'], entry['item'], entry['query']

    def item_sampling(self, negNum:int, start:int = 0, end:int = 0):
        easyNeg = int(negNum * 0.9)
        hardNeg = negNum-easyNeg
        randcut = int(self.itemNum /3 )
        rand_pos = random.randint(randcut,self.itemNum-randcut)
        # easyNeg = negNum
        for idx, entry in enumerate(tqdm(self.dataSave,
                                         desc='item sampling',
                                         total=len(self.dataSave),
                                         ncols=117, unit_scale=True)):
            neg_items = PersonalizedNeg(self.u2iMatrix,
                               entry['user'] - len(self.itemMap),
                               start, end, easyNeg)
            att = self.item2attribute[entry['item']]
            X = np.abs(self.all_attributes - att)/(1e8)
            x = np.exp(np.sum(X ** 2, axis = -1))
            temp = (x.argsort()).tolist() # the most similar items. BUT IT MIGHT not be resonable to selecy the ons with highest similarity.
            res = list(filter(lambda x: x not in neg_items, temp[rand_pos:]))
            neg_items.extend(res[:hardNeg])
            self.dataSave[idx]['item_neg'] = torch.tensor(neg_items, dtype=torch.long)


    def non_personalized_sampling(self, neg_num, start=0, end=None):
        """ Negative sampling without personalization. """
        for idx, entry in enumerate(tqdm(self.dataSave,
                                         desc='item sampling',
                                         total=len(self.dataSave),
                                         ncols=117, unit_scale=True)):
            negatives = []
            for _ in range(neg_num):
                j = np.random.randint(start, end)
                while j == entry['item'] or j in negatives:
                    j = np.random.randint(start, end)
                negatives.append(j)
            self.data[idx]['item_neg'] = torch.tensor(negatives, dtype=torch.long)

    @staticmethod
    def init(full_df: pd.DataFrame):
        users = full_df['userID'].unique()
        items = full_df['asin'].unique()
        item_map = dict(zip(items, range(len(items))))
        attribute_max_len = max(map(lambda x: len(eval(x)), full_df['attrWords']))
        query_max_length = max(map(lambda x: len(eval(x)), full_df['queryWords']))
        return users, item_map, query_max_length,attribute_max_len

