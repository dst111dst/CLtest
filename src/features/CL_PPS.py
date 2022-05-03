import torch.nn as nn
import torch
from tqdm import tqdm
from src.features.AugmentedDataset import nCr
from src.tools.Loss import *

class Trainer(nn.Module):
    def __init__(self, word_num: int, entity_num: int, embedding_size: int, head_num: int,
                 max_seq_length: int, hidden_size :int,
        n_views: int,cuda_condition: bool = False):
        super(Trainer, self).__init__()
        self.device = torch.device("cuda" if cuda_condition else "cpu")
        self.total_augmentaion_pairs = nCr(n_views, 2)
        self.max_seq_length = max_seq_length
        self.hidden_size = hidden_size
        self.entity_embed = nn.Embedding(entity_num, embedding_size)
        self.word_embed = nn.Embedding(word_num + 1, embedding_size, padding_idx=word_num)
        self.word_encode = nn.MultiheadAttention(embedding_size, head_num, batch_first=True)
        # projection head for contrastive learn task
        self.project_layer = nn.Sequential(nn.Linear( max_seq_length * hidden_size, \
                                                  512, bias=False), nn.BatchNorm1d(512), nn.ReLU(inplace=True),
                                        nn.Linear(512,hidden_size, bias=True))
        if self.cuda_condition:
            self.projection_layer.cuda()

    def sent_encode(self, sentence):
        sent_embed  = self.word_embed(sentence)
        sent_encode = self.word_encode(sent_embed, sent_embed, sent_embed)[0]
        sent_encode = sent_encode.mean(dim=1)
        return sent_encode


class contrastiveSearch(nn.Module):
    def __init__(self, word_num: int, entity_num: int, embedding_size: int, head_num: int,\
                 max_seq_length: int, hidden_size :int,\
                 n_views: int,  cuda_condition: bool = False,
                 search_weight: float = 0.9, cl_weight: float = 0.1,\
        temprature: float = 1.0, num_layers: int = 3, dropout: float = 0.5):

        super().__init__()
        self.device = torch.device("cuda" if cuda_condition else "cpu")
        self.cuda_condition = cuda_condition
        self.total_augmentaion_pairs = nCr(n_views, 2)
        self.max_seq_length = max_seq_length
        self.hidden_size = hidden_size
        self.entity_embed = nn.Embedding(entity_num, embedding_size)
        self.word_embed = nn.Embedding(word_num + 1, embedding_size, padding_idx=word_num)
        self.word_encode = nn.MultiheadAttention(embedding_size, head_num, batch_first=True)
        # projection head for contrastive learn task
        self.project_layer = nn.Sequential(nn.Linear(max_seq_length * hidden_size, 512, bias=False), nn.BatchNorm1d(512), nn.ReLU(inplace=True),
                                           nn.Linear(512, hidden_size, bias=True))
        if self.cuda_condition:
            self.projection_layer.cuda()

        self.MLP_embed = nn.Embedding(entity_num,embedding_size * (2 ** (num_layers - 1)))
        MLP_modules = []
        # Initialize MLP layers.
        for i in range(num_layers):
            input_size = embedding_size * (2 ** (num_layers - i))
            MLP_modules.append(nn.Dropout(p=dropout))
            MLP_modules.append(nn.Linear(input_size, input_size // 2))
            MLP_modules.append(nn.ReLU())
        self.MLP_layers = nn.Sequential(*MLP_modules)

        # for NCF only
        self.prediction = nn.Linear(embedding_size * 2, 1)

        # CL loss
        self.temperature = temprature
        self.cl_criterion = NCELoss(self.temperature, self.device)
        self.search_weight  = search_weight
        self.cl_weight = cl_weight

    def sent_encode(self, sentence):
        sent_embed  = self.word_embed(sentence)
        sent_encode = self.word_encode(sent_embed, sent_embed, sent_embed)[0]
        sent_encode = sent_encode.mean(dim=1)
        return sent_encode

    def forward(self, users, items, query_words, contras_seq,
                review_words = None,
                mode= 'train', neg_items = None):
        """
        :param users: [batch, ]
        :param items: [batch, ]
        :param query_words: [batch, num_query_words]
        :param review_words: [batch, num_sent_words]
        :param mode: ('train', 'test', 'out_embedding')
        :param neg_items: [batch, n]
        """
        if mode == 'output_embedding':
            item_embed_gmf = self.entity_embed(items)
            return item_embed_gmf

        query_encode = self.sent_encode(query_words)
        user_embed_gmf = self.entity_embed(users)
        user_embed_mlp = self.MLP_embed(users)
        personalized = query_encode + 0.5 * user_embed_gmf

        if mode == 'test':
            return personalized

        if mode == 'train':
            # items
            item_embed_gmf = self.entity_embed(items)
            item_embed_mlp = self.MLP_embed(items)

            neg_item_embed_gmf = self.entity_embed(neg_items)
            neg_item_embed_mlp = self.MLP_embed(neg_items)

            # where ncf begins
            fusion_gmf = user_embed_gmf * item_embed_gmf
            fusion_mlp = self.MLP_layers(torch.cat([user_embed_mlp, item_embed_mlp], dim=-1))
            neg_fusion_gmf = user_embed_gmf.unsqueeze(dim=1) * neg_item_embed_gmf
            neg_fusion_mlp = self.MLP_layers((torch.cat(
                [torch.stack([user_embed_mlp for _ in range(neg_items.size()[1])], dim=1),
                 neg_item_embed_mlp], dim=-1)))

            fusion = self.prediction(torch.cat([fusion_gmf, fusion_mlp], dim=-1))
            neg_fusion = self.prediction(torch.cat([neg_fusion_gmf, neg_fusion_mlp], dim=-1))

            cf_loss = ncf_bce_loss(fusion, neg_fusion)
            search_loss = nce_loss(personalized, item_embed_gmf, neg_item_embed_gmf)

            # for contras:
            joint_loss = self.search_weight * search_loss + cf_loss # not involves cf?

            cl_loss = []
            for pair in contras_seq:
                cl_item_embed_gmf = self.entity_embed(pair)
                cl_loss.append(nce_loss(personalized, item_embed_gmf, cl_item_embed_gmf))
            for c in cl_loss:
                joint_loss += c* self.cl_weight
            return joint_loss
