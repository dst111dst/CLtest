import os
import time
from argparse import ArgumentParser
from ParseAmazon import *
from src.tools.Params import ParserParams

if __name__ == '__main__':
    parser = ArgumentParser()
    ParserParams(parser)
    args = parser.parse_args()
    random.seed(args.seed)
    start = time.time()
    if not os.path.exists(args.data_path):
        os.makedirs(args.data_path)
    if not os.path.exists(args.processed_path):
        os.makedirs(args.processed_path)

    args.dataset = 'Cell_Phones_and_Accessories'
    Metapath = os.path.join(args.data_path, "meta_{}.json.gz".format(args.dataset))
    Reviewpath = os.path.join(args.data_path, "reviews_{}_5.json.gz".format(args.dataset))
    ReviewDf = ParseReviews(Reviewpath)
    Categories, also_viewed, Attributes = ParseMeta(Metapath)

    Temp_df, word_dict = ParseWords(ReviewDf,args.word_count,categories=Categories,attributes=Attributes,defaultQuery=args.dataset.split('_'))
    # raise ValueError
    Temp_df = Temp_df.drop(['reviewerName', 'reviewTime', 'summary', 'overall', 'helpful'], axis=1)
    df = ReviewMap(Temp_df)
    df = SplitData(df,ratio=0.7)
    # write processed results to disk
    processed_path = os.path.join(args.processed_path, args.dataset)
    if not os.path.exists(processed_path):
        os.makedirs(processed_path)
    json.dump(word_dict, open(os.path.join(processed_path, 'word_dict.json'), 'w'))
    df = DelDupRev(df, word_dict)  # remove the reviews from test set
    csv_path = os.path.join(processed_path, 'full.csv')
    df.to_csv(csv_path, index=False,encoding='utf-8')
    print("The number of {} users is {:d}; items is {:d}; feedbacks is {:d}; words is {:d}.".format(
        args.dataset, len(df.reviewerID.unique()), len(df.asin.unique()), len(df), len(word_dict)),
        "costs:", time.strftime("%H: %M: %S", time.gmtime(time.time() - start)))



