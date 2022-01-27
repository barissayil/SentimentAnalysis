import torch.nn as nn
from torch.utils.data import DataLoader

from dataset import SSTDataset
from arguments import args
from analyzer import Analyzer


if __name__ == "__main__":

    # Initialize analyzer.
    analyzer = Analyzer(will_train=False, args=args)

    # Set citerion, which takes as input logits of positive class and computes binary cross-entropy.
    criterion = nn.BCEWithLogitsLoss()

    # Initialize validation set and loader.
    val_set = SSTDataset(
        filename="data/dev.tsv", maxlen=args.maxlen_val, tokenizer=analyzer.tokenizer
    )
    val_loader = DataLoader(
        dataset=val_set, batch_size=args.batch_size, num_workers=args.num_threads
    )

    # Evaluate analyzer and get accuracy + loss.
    val_accuracy, val_loss = analyzer.evaluate(
        val_loader=val_loader, criterion=criterion
    )

    # Display accuracy and loss.
    print(f"Validation Accuracy : {val_accuracy}, Validation Loss : {val_loss}")
