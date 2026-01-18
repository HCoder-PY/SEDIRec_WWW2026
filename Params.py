import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="Run STLLMRec.")
    parser.add_argument('--seed', type=int, default=2024,
                        help='Random seed.')
    parser.add_argument('--dataset', nargs='?', default='dbbook2014',
                        help='Choose a dataset from {dbbook2014, book-crossing, ml1m}')
    parser.add_argument("--gpu_id", type=int, default=4, help="gpu id")
    parser.add_argument('--norm', type=bool, default=True)

    args = parser.parse_args()
    return args

args = parse_args()
