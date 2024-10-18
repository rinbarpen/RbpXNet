import sys
from argparse import ArgumentParser
from utils.utils import print_model_info, summary_model_info

def parse_args():
    parser = ArgumentParser('SummaryModel')
    parser.add_argument('-m', '--model', required=True, help='Summary Model Information')
    parser.add_argument('-b', '--brief', action='store_true', default=True, help='Summary Model Information With Brief Description')
    parser.add_argument('--input_size', type=str, help='Only use for summary models in detail')
    args = parser.parse_args()
    if not args.brief:
        if not args.input_size:
            raise ValueError('Input size must be specified in detail')
        else:
            args.input_size = tuple(args.input_size.split(','))

    return args

if __name__ == '__main__':
    args = parse_args()
    if args.brief:
        print_model_info(args.model, sys.stdout)
    else:
        summary_model_info(args.model, input_size=args.input_size)
