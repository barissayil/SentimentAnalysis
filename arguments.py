from argparse import ArgumentParser

parser = ArgumentParser()

parser.add_argument('--maxlen_train', type=int, default=30)
parser.add_argument('--maxlen_val', type=int, default=50)
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--lr', type=float, default=2e-5)
parser.add_argument('--num_eps', type=int, default=2)
parser.add_argument('--num_threads', type=int, default=1)
parser.add_argument('--output_dir', type=str, default='my_model')
parser.add_argument('--model_name_or_path', type=str, default='')

args = parser.parse_args()