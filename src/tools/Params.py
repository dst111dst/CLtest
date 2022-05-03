def ParserParams(parser):
    # ------------------------------------Dataset Parameters-------------------- #
    parser.add_argument('--data_path',
                        type=str, default='/Users/tt/Downloads/cl4pps/data/raw/',
                        help="raw downloaded path")
    parser.add_argument('--processed_path',
                        type=str, default='/Users/tt/Downloads/cl4pps/data/processed/',
                        help="after processed path")
    parser.add_argument('--save_path',
                        type=str, default='/Users/tt/Downloads/cl4pps/models/',
                        help="after processed path")
    parser.add_argument('--dataset',
                        type=str, default='Musical_Instruments',
                        help="Want to compare with other models ran yesterday..the dataset name of your model")
    parser.add_argument('--model',
                        type=str,
                        help="the model name (for AEM and CFSearch only)")
    # ------------------------------------Process Parameters-------------------- #
    parser.add_argument('--seed',
                        type=int, default=11,
                        help="for code reproduction")
    parser.add_argument('--word_count',
                        type=int, default=10,
                        help="remove the words number less than count")
    parser.add_argument("--doc2vec_size",
                        type=int,
                        default=512,
                        help="doc2vec model embedding dimension")
    parser.add_argument('--candidate',
                        type=int, default=100,
                        help="rank results on 100 candidate items")
    # ------------------------------------Experiment Setups -------------------- #
    parser.add_argument('--debug',
                        default=True,
                        action='store_true',
                        help="enable debug")
    parser.add_argument('--gpu',
                        default='0',
                        help="using device")
    parser.add_argument('--worker_num',
                        default=4,
                        type=int,
                        help='number of workers for data loading')
    parser.add_argument('--top_k',
                        default=10,
                        type=int,
                        help='truncated at top_k products')
    parser.add_argument('--max_query_len',
                        default=20,
                        type=int,
                        help='max length for each query')
    parser.add_argument('--max_sent_len',
                        default=100,
                        type=int,
                        help='max length for each review')
    parser.add_argument('--embedding_size',
                        default=128,
                        type=int,
                        help="embedding size for possibly word, user and item")
    parser.add_argument('--lr',
                        default=1e-3,
                        type=float,
                        help='learning rate')
    parser.add_argument('--regularization',
                        default=1e-3,
                        type=float,
                        help='regularization factor')
    parser.add_argument('--batch_size',
                        default=256,
                        type=int,
                        help='batch size for training')
    parser.add_argument('--neg_sample_num',
                        default=5,
                        type=int,
                        help='negative sample number')
    parser.add_argument('--epochs',
                        default=30,
                        type=int,
                        help="training epochs")
    parser.add_argument('--augment_type',
                        default='reorder',
                        type=str,
                        help="Basic augmentation type")
    parser.add_argument('--augment_threshold', default=4, type=int,
                        help="control augmentations on short and long sequences.\
                            default:-1, means all augmentations types are allowed for all sequences.\
                            For sequence length < augment_threshold: Insert, and Substitute methods are allowed \
                            For sequence length > augment_threshold: Crop, Reorder, Substitute, and Mask \
                            are allowed.")
    parser.add_argument('--augment_type_for_short', default='SIM', type=str, \
                        help="data augmentation types for short sequences. Chosen from: \
                            SI, SIM, SIR, SIC, SIMR, SIMC, SIRC, SIMRC.")
    parser.add_argument('--temperature', default=1.0, type=float,
                        help='softmax temperature (default:  1.0) - not studied.')
    parser.add_argument('--max_seq_length', default=50, type=int)
    parser.add_argument("--hidden_size", type=int, default=64, help="hidden size of transformer model")


from tqdm import tqdm
def training_progress(loader, epoch, epochs, loss, debug):
    return tqdm(loader, desc="Running Epoch {:03d}/{:03d}".format(epoch + 1, epochs),
                ncols=117, unit=' steps', unit_scale=True,
                postfix={"loss": "{:.3f}".format(float(loss))}) if debug else loader

def testing_progress(loader, epoch, epochs, debug):
    return tqdm(loader, desc="Testing Epoch {:03d}/{:03d}".format(epoch + 1, epochs),
                ncols=117, unit=' users', unit_scale=True) if debug else loader

def building_progress(df, debug, desc='building'):
    return tqdm(df.iterrows(), desc=desc, total=len(df),
                ncols=117, unit=' entries', unit_scale=True) if debug else df.iterrows()