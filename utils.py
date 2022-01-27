import torch

def get_accuracy_from_logits(logits, labels):
	# Convert logits to probabilties
	probabilties = torch.sigmoid(logits.unsqueeze(-1))
	# Convert probabilities to predictions (1: positive, 0: negative)
	predictions = (probabilties > 0.5).long().squeeze()
	# Calculate qnd return accuracy
	return (predictions == labels).float().mean()