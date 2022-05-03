import json
import numpy as np
import torch
import os
from src.data.AmazonDataset import AmazonDataset
from src.tools.Metrics import hit, mrr, ndcg
from src.features.AugmentedDataset import PPSWithContrastiveLearningDataset

def chunk_test(start, end, interval=512):
    count   = start
    all_ids = []
    while count < end:
        chunk_ids = [count + i for i in range(interval) if count + i < end]
        count += len(chunk_ids)
        all_ids.append(torch.tensor(chunk_ids, dtype=torch.long))
        # all_ids.append(torch.tensor(chunk_ids, dtype=torch.long).cuda())
    return all_ids


def eval_candidates(model, test_dataset, test_loader, top_k, num_cands,save_path = None):
    """ Evaluate on num_cands items. """
    with torch.no_grad():
        Hr, Mrr, Ndcg = [], [], []
        for _, (user, item, query) in enumerate(test_loader):
            assert len(user) == num_cands and all(user == user[0])
            # user        = user.cuda()
            # item        = item.cuda()
            # query       = query.cuda()
            item_embed  = model(None, item, None, mode='output_embedding')
            pred        = model(user, None, query, mode='test')
            scores      = (pred * item_embed).sum(dim=-1)

            _, indices  = scores.topk(top_k, largest=True)
            indices     = indices.cpu().numpy().tolist()

            # we test with abuse use of ground-truth index
            Hr.append(hit(0, indices))
            Mrr.append(mrr(0, indices))
            Ndcg.append(ndcg(0, indices))

        return np.mean(Hr), np.mean(Mrr), np.mean(Ndcg)


def evaluate(model, test_dataset: AmazonDataset, test_loader, top_k,save_path = None):
    with torch.no_grad():
        Hr, Mrr, Ndcg       = [], [], []
        chunk_items_ids     = chunk_test(0, len(test_dataset.itemMap))
        chunk_items_embed   = []
        for items in chunk_items_ids:
            chunk_items_embed.append(model(None, items, None, mode='output_embedding'))
        user_buy = test_dataset.userBuy
        if save_path is not None:
            embed_path = os.path.join(save_path, 'embed.npy')
            embed_handle = embed_path
            dict_path = os.path.join(save_path,'search_info.json' )
            rank_results = dict()
            result_handle = dict_path
        else:
            embed_handle = './embed.npy'
            rank_results = dict()
            result_handle = './search_info.json'
        item_map_reverse = {v: k for k, v in test_dataset.itemMap.items()}

        for _, (user, item, query) in enumerate(test_loader):
            assert len(item) == 1 and len(query) == 1
            item    = item.item()
            # user    = user.cuda()
            # query   = query.cuda()
            # ---------rank all--------- #
            pred    = model(user, None, query, mode='test')
            scores  = []
            for item_embeds in chunk_items_embed:
                scores.append(torch.sum(pred * item_embeds, dim=-1))
            scores  = torch.cat(scores)

            _, ranking_list = scores.sort(descending=False)
            ranking_list    = ranking_list.tolist()
            bought      = user_buy[user.cpu().item()]
            return_list = []
            while len(return_list) < top_k:
                if len(ranking_list) == 0:
                    break
                candidate_item = ranking_list.pop()
                if not model.__class__.__name__ == 'VanillaSearch':
                    if candidate_item not in bought or candidate_item == item:
                        return_list.append(candidate_item)
                else:
                    return_list.append(candidate_item)
            Hr.append(hit(item, return_list))
            Mrr.append(mrr(item, return_list))
            Ndcg.append(ndcg(item, return_list))

            rank_results[user.cpu().item()] = {
                'ranking_list': return_list,
                'ndcg': ndcg(item, return_list)
            }
        with open(result_handle, 'w') as fd:
            json.dump(rank_results, fd)
        np.save(embed_handle, model.entity_embed.weight.cpu().numpy())
    return np.mean(Hr), np.mean(Mrr), np.mean(Ndcg)

def contrasEval(model, test_dataset:PPSWithContrastiveLearningDataset , test_loader, top_k,save_path = None):
    with torch.no_grad():
        Hr, Mrr, Ndcg       = [], [], []
        chunk_items_ids     = chunk_test(0, len(test_dataset.itemMap))
        chunk_items_embed   = []
        for items in chunk_items_ids:
            chunk_items_embed.append(model(None, items, None,None, mode='output_embedding'))
        user_buy = test_dataset.userBuy
        if save_path is not None:
            embed_path = os.path.join(save_path, 'embed.npy')
            embed_handle = embed_path
            dict_path = os.path.join(save_path,'search_info.json' )
            rank_results = dict()
            result_handle = dict_path
        else:
            embed_handle = './embed.npy'
            rank_results = dict()
            result_handle = './search_info.json'

        for _, (user, item, query) in enumerate(test_loader):
            assert len(item) == 1 and len(query) == 1
            item    = item.item()
            pred    = model(user, None, query, None,mode='test')
            scores  = []
            for item_embeds in chunk_items_embed:
                scores.append(torch.sum(pred * item_embeds, dim=-1))
            scores  = torch.cat(scores)
            _, ranking_list = scores.sort(descending=False)
            ranking_list    = ranking_list.tolist()
            return_list = []
            while len(return_list) < top_k:
                if len(ranking_list) == 0:
                    break
                candidate_item = ranking_list.pop()
                return_list.append(candidate_item)
            Hr.append(hit(item, return_list))
            Mrr.append(mrr(item, return_list))
            Ndcg.append(ndcg(item, return_list))

            rank_results[user.cpu().item()] = {
                'ranking_list': return_list,
                'ndcg': ndcg(item, return_list)
            }
        with open(result_handle, 'w') as fd:
            json.dump(rank_results, fd)
        np.save(embed_handle, model.entity_embed.weight.cpu().numpy())
    return np.mean(Hr), np.mean(Mrr), np.mean(Ndcg)