import argparse

from arguments import args

def test_arguments():
  assert isinstance(args, argparse.Namespace)