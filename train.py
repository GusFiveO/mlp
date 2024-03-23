#! /usr/bin/env python3

from neural_network import NeuralNetwork
from utils import load_csv
import pandas as pd

def train(df: pd.DataFrame):
	targets = df.pop('2')
	print(df)
	print(targets)
	model = NeuralNetwork(df, targets, 10, 0.1, [2], "uniform")
	print(model)
	model.forward_propagation()

if __name__ == "__main__":
	# df = load_csv('./data_mlp.csv')
	df = pd.read_csv('./data_mlp.csv', nrows=3)
	if df is None:
		exit()
	train(df)