from argparse import ArgumentParser

parser = ArgumentParser()

parser.add_argument('-maxlen', type=int, default=25)
parser.add_argument('-batch_size', type=int, default=32)
parser.add_argument('-lr', type=float, default=2e-5)
parser.add_argument('-num_eps', type=int, default=5)
parser.add_argument('-num_threads', type=int, default=1)
parser.add_argument('-model_type', type=str, default='bert')
parser.add_argument('-model_name', type=str, default='bert-base-uncased')

args = parser.parse_args()