import torch
from torch.utils.data import Dataset
import math
from src.features.Augmentation import *
import random
from tqdm import tqdm
from src.tools.Params import building_progress,testing_progress,training_progress
import scipy.sparse as sp
from typing import Dict

from src.tools.Sample import *


def RecNegSample(item_set, item_size: int) -> int:
    item = random.randint(1, item_size - 1)
    while item in item_set:
        item = random.randint(1, item_size - 1)
    return item

def nCr(n: int,r: int) -> int:
    f = math.factorial
    return f(n) // f(r) // f(n-r)


class PPSWithContrastiveLearningDataset(Dataset):
    '''
        def __init__(self, input_df:pd.DataFrame,
                 users:list, item_map: Dict[str,int],
                 TotalWords:int, QueryMax:int,
                 SentenceMax:int, AttrMax:int,
                 Mode:str, isDebug:bool, userBuy:List = None):
        self.Mode = Mode
        self.userSeq = users
        self.itemMap = item_map
        self.wordNum = TotalWords
        self.userBuy = userBuy
        self.userNum = len(users)
        self.itemNum = len(item_map)
        self.dataSave = []
    '''
    def __init__(self, input_df:pd.DataFrame,
                 users:list, item_map: Dict[str,int],
                 TotalWords:int, QueryMax:int,
                 SentenceMax:int, AttrMax:int,
                 Mode:str, simModel, isDebug:bool, augment_type: str,\
                 tao: float = 0.8, gamma: float =0.5, beta: float =0.5, \
                 insert_rate: float =0.4,max_insert_num_per_pos: int =3,\
                 substitute_rate: float =0.3,augment_threshold:int = -1,augment_type_for_short:str = 'SIM', \
                 n_views=5, userBuy:Dict = None):
        self.Mode = Mode
        self.itemMap = item_map
        self.wordNum = TotalWords
        self.userBuy = userBuy
        self.userNum = len(users)
        self.itemNum = len(item_map)
        self.dataSave = []

        self.augment_type = augment_type
        self.user_seq = users
        self.test_neg_items = None
        self.max_len = len(item_map)
        self.similarity_model = simModel

        self.augmentations = {'crop': Crop(tao=tao),
                              'mask': Mask(gamma=gamma),
                              'reorder': Reorder(beta=beta),
                              'substitute': Substitute(self.similarity_model,substitute_rate),
                              'insert': Insert(self.similarity_model,
                                               insert_rate=insert_rate,
                                               max_insert_num_per_pos=max_insert_num_per_pos),
                              'random': Random(tao=tao, gamma=gamma,
                                               beta=beta, item_similarity_model=self.similarity_model,
                                               insert_rate=insert_rate,
                                               max_insert_num_per_pos=max_insert_num_per_pos,
                                               substitute_rate = substitute_rate,
                                               augment_threshold= augment_threshold,
                                               augment_type_for_short=augment_type_for_short),
                              'combinatorial_enumerate': CombinatorialEnumerate(tao=tao, gamma=gamma,
                                                                                beta=beta,
                                                                                item_similarity_model=self.similarity_model,
                                                                                insert_rate=insert_rate,
                                                                                max_insert_num_per_pos=max_insert_num_per_pos,
                                                                                substitute_rate = substitute_rate,
                                                                                n_views=n_views)
                              }
        self.base_transform = self.augmentations[augment_type]
        # number of augmentations for each sequences, current support two
        self.n_views = n_views

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

    def _one_pair_data_augmentation(self, input_ids):
        '''
        provides two positive samples given one sequence
        :params: input_ids -> List contains the item-id
        '''
        augmented_seqs = []
        for i in range(2):
            augmented_input_ids = self.base_transform(input_ids)
            pad_len = self.max_len - len(augmented_input_ids)
            augmented_input_ids = [0] * pad_len + augmented_input_ids
            augmented_input_ids = augmented_input_ids[-self.max_len:]
            assert len(augmented_input_ids) == self.max_len
            cur_tensors = torch.tensor(augmented_input_ids, dtype=torch.long)
            augmented_seqs.append(cur_tensors)
        return augmented_seqs

    def _common_res(self,index):
        entry = self.dataSave[index]
        if self.Mode == 'train':
            return entry['user'], entry['item'], entry['query'], \
                   entry['word'], entry['item_neg']
        else:
            return entry['user'], entry['item'], entry['query']

    def __len__(self):
        '''
        consider n_view of a single sequence as one sample
        '''
        return len(self.user_seq)

    def __getitem__(self, index):
        entry = self.dataSave[index]
        item_id = self.userBuy[int(entry['user'])]  # doubt about it.
        if self.Mode == 'train':
            total_augmentaion_pairs = nCr(self.n_views, 2)
            # cf_tensors_list = []
            # for i in range(total_augmentaion_pairs):
            #     cf_tensors_list.append(self._one_pair_data_augmentation(item_id))
            return (self._common_res(index), self._one_pair_data_augmentation(item_id))
        else:
            return self._common_res(index)

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
            x = (np.sum(X ** 2, axis = -1))
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
            self.dataSave[idx]['item_neg'] = torch.tensor(negatives, dtype=torch.long)

    @staticmethod
    def init(full_df: pd.DataFrame):
        users = full_df['userID'].unique()
        items = full_df['asin'].unique()
        item_map = dict(zip(items, range(len(items))))
        attribute_max_len = max(map(lambda x: len(eval(x)), full_df['attrWords']))
        query_max_length = max(map(lambda x: len(eval(x)), full_df['queryWords']))
        return users, item_map, query_max_length,attribute_max_len

if __name__ == '__main__':
    from argparse import ArgumentParser
    from torch.utils.data import DataLoader
    from src.tools.Params import *
    from src.tools.LoadProcessedData import data_preparation

    parser = ArgumentParser()
    ParserParams(parser)
    # ------------------------------------Experiment Setup------------------------------------ #
    parser.add_argument('--head_num',
                        default=4,
                        type=int,
                        help='the number of heads used in multi-head self-attention layer')
    parser.add_argument('--conv_num',
                        default=2,
                        type=int,
                        help='the number of convolution layers')

    args = parser.parse_args()
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    train_df, test_df, full_df, word_dict = data_preparation(args)
    users, item_map, query_max_length, attribute_max_len = PPSWithContrastiveLearningDataset.init(full_df)
    query_max_length = min(query_max_length, args.max_query_len)
    sv_path = '/Users/tt/Downloads/cl4pps/src/models/similarity.pkl'
    feature_based_similarity = FeatureBasedSimilarity(similarity_path=sv_path)
    n_views = 5
    # --------------------------------------Data Loaders-------------------------------------- #
    train_dataset = PPSWithContrastiveLearningDataset(train_df, users, item_map,
                                  len(word_dict), query_max_length, args.max_sent_len, attribute_max_len,
                                  'train', feature_based_similarity, args.debug,args.augment_type,\
                                                      args.augment_threshold,args.augment_type_for_short)
    test_dataset = PPSWithContrastiveLearningDataset(test_df, users, item_map,
                                 len(word_dict), query_max_length, args.max_sent_len, attribute_max_len,
                                 'test',feature_based_similarity, args.debug,args.augment_type,\
                                args.augment_threshold,args.augment_type_for_short,userBuy=train_dataset.userBuy)

    train_loader = DataLoader(train_dataset, drop_last=False, batch_size=args.batch_size,
                              shuffle=True, num_workers=args.worker_num)
    train_loader.dataset.item_sampling(negNum = args.neg_sample_num,start = 0, end=len(item_map))
    rec_cf_data_iter = tqdm(enumerate(train_loader), total = len(train_loader))
    