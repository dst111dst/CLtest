import os
import numpy as np
from argparse import ArgumentParser
from src.features.Augmentation import *
from torch.utils.data import DataLoader
from src.features.AugmentedDataset import PPSWithContrastiveLearningDataset
from src.features.CL_PPS import *
from src.models.Evaluate import contrasEval

from src.tools.Metrics import display
from src.tools.Params import *
from src.tools.LoadProcessedData import data_preparation

def run(model_name: str, args):
    torch.manual_seed(args.seed)
    train_df, test_df, full_df, word_dict = data_preparation(args)
    users, item_map, query_max_length,attribute_max_len = PPSWithContrastiveLearningDataset.init(full_df)
    query_max_length = min(query_max_length, args.max_query_len)
    sv_path = '/Users/tt/Downloads/cl4pps/src/models/similarity.pkl'
    feature_based_similarity = FeatureBasedSimilarity(similarity_path=sv_path)
    n_views = 5
    train_dataset = PPSWithContrastiveLearningDataset(train_df, users, item_map,
                                                      len(word_dict), query_max_length, args.max_sent_len,
                                                      attribute_max_len,
                                                      'train', feature_based_similarity, args.debug, args.augment_type, \
                                                      args.augment_threshold, args.augment_type_for_short)
    test_dataset = PPSWithContrastiveLearningDataset(test_df, users, item_map,
                                                     len(word_dict), query_max_length, args.max_sent_len,
                                                     attribute_max_len,
                                                     'test', feature_based_similarity, args.debug, args.augment_type, \
                                                     args.augment_threshold, args.augment_type_for_short,
                                                     userBuy=train_dataset.userBuy)

    train_loader = DataLoader(train_dataset, drop_last=False, batch_size=args.batch_size,
                              shuffle=True, num_workers=args.worker_num)
    test_loader = DataLoader(test_dataset, drop_last=False, batch_size=1,
                             shuffle=False, num_workers=0)  # for whole set

    """
    self, word_num: int, entity_num: int, embedding_size: int, head_num: int,
                 max_seq_length: int, hidden_size :int,
        n_views: int,cuda_condition: bool = False, search_weight: float = 0.9, cl_weight: float = 0.3,\
        temprature: float = 1.0, num_layers: int = 3, dropout: float = 0.5
    """
    model = contrastiveSearch(len(word_dict),
                          len(users) + len(item_map), args.embedding_size, args.head_num,
                          args.max_seq_length, args.hidden_size,n_views)
    optim = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.regularization)
    # ------------------------------------Train------------------------------------ #
    loss = 0
    for epoch in range(args.epochs):
        model.train()
        epoch_loss = step = 0
        train_loader.dataset.item_sampling(negNum = args.neg_sample_num,start = 0, end=len(item_map))
        # progress = training_progress(train_loader, epoch, args.epochs, loss, args.debug)
        rec_cf_data_iter = tqdm(enumerate(train_loader), total=len(train_loader))
        for i, (rec_batch, cf_batch) in rec_cf_data_iter:
            model.zero_grad()
            users, items, query_words, words, items_neg = rec_batch
            loss = model(users, items, query_words,cf_batch, words, 'train', items_neg)
            if args.debug:
                rec_cf_data_iter.set_postfix({"loss": "{:.3f}".format(float(loss))})
            epoch_loss += loss
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.)
            optim.step()
            step += 1

        model.eval()
        temp_path = args.save_path + args.dataset + '/' + model_name + '/'
        if not os.path.exists(temp_path):
            os.makedirs(temp_path)
        # for saving the evaluate results.If we just use other models for compare, then
        # we won't save for their models.
        # Hr, Mrr, Ndcg = eval_candidates(model, test_dataset,
        #                                 testing_progress(test_loader, epoch, args.epochs, args.debug),
        #                                 args.top_k, args.candidate)
        Hr, Mrr, Ndcg =contrasEval(model, test_dataset,
                                 testing_progress(test_loader, epoch, args.epochs, args.debug),
                                 args.top_k)
        display(epoch, args.epochs, epoch_loss / step, Hr, Mrr, Ndcg)


if __name__ == '__main__':
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
    run('CLsearch',args)
