import torch
from transformers import AutoTokenizer, AutoConfig
from tqdm import tqdm

from modeling import (
    BertForSentimentClassification,
    AlbertForSentimentClassification,
    DistilBertForSentimentClassification,
)
from utils import get_accuracy_from_logits


class Analyzer:
    def __init__(self, will_train, args):

        # If no model name/path is given, use mine/BERT depending on task.
        if args.model_name_or_path is None:
            if will_train:
                args.model_name_or_path = "bert-base-uncased"
            else:
                args.model_name_or_path = "barissayil/bert-sentiment-analysis-sst"

        # Set up configuration.
        self.config = AutoConfig.from_pretrained(args.model_name_or_path)

        # Create the model with the given configuration.
        if self.config.model_type == "bert":
            self.model = BertForSentimentClassification.from_pretrained(
                args.model_name_or_path
            )
        elif self.config.model_type == "albert":
            self.model = AlbertForSentimentClassification.from_pretrained(
                args.model_name_or_path
            )
        elif self.config.model_type == "distilbert":
            self.model = DistilBertForSentimentClassification.from_pretrained(
                args.model_name_or_path
            )
        else:
            raise ValueError("This transformer model is not supported yet.")

        # Set up device as GPU if available, otherwise CPU.
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        # Put model to device.
        self.model = self.model.to(self.device)

        # Set model to evaluation mode.
        self.model.eval()

        # Initialize tokenizer for the desired transformer model
        self.tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)

        # Set output directory.
        self.output_dir = args.output_dir

    # Evaluates analyzer.
    def evaluate(self, val_loader, criterion):
        # Set model to evaluation mode.
        self.model.eval()
        # Initialize batch accuracy summation, loss, and number of batches.
        batch_accuracy_summation, loss, num_batches = 0, 0, 0
        # Don't track gradient.
        with torch.no_grad():
            # Go through validation set in batches.
            for input_ids, attention_mask, labels in tqdm(
                val_loader, desc="Evaluating"
            ):
                # Put input IDs, attention mask, and labels to device.
                input_ids, attention_mask, labels = (
                    input_ids.to(self.device),
                    attention_mask.to(self.device),
                    labels.to(self.device),
                )
                # Get logits.
                logits = self.model(input_ids=input_ids, attention_mask=attention_mask)
                # Get batch accuracy and add it.
                batch_accuracy_summation += get_accuracy_from_logits(logits, labels)
                # Get batch loss and add it.
                loss += criterion(logits.squeeze(-1), labels.float()).item()
                # Increment num_batches.
                num_batches += 1
        # Calculate accuracy.
        accuracy = batch_accuracy_summation / num_batches
        # Return accuracy and loss.
        return accuracy.item(), loss

    # Trains analyzer for one epoch.
    def train(self, train_loader, optimizer, criterion):
        # Set model to training mode.
        self.model.train()
        # Go through training set in batches.
        for input_ids, attention_mask, labels in tqdm(
            iterable=train_loader, desc="Training"
        ):
            # Reset gradient
            optimizer.zero_grad()
            # Put input IDs, attention mask, and labels to device
            input_ids, attention_mask, labels = (
                input_ids.to(self.device),
                attention_mask.to(self.device),
                labels.to(self.device),
            )
            # Get logits.
            logits = self.model(input_ids=input_ids, attention_mask=attention_mask)
            # Get loss.
            loss = criterion(input=logits.squeeze(-1), target=labels.float())
            # Backpropagate the loss.
            loss.backward()
            # Optimize the model.
            optimizer.step()

    # Saves analyzer.
    def save(self):
        # Save model.
        self.model.save_pretrained(save_directory=f"models/{self.output_dir}/")
        # Save configuration.
        self.config.save_pretrained(save_directory=f"models/{self.output_dir}/")
        # Save tokenizer.
        self.tokenizer.save_pretrained(save_directory=f"models/{self.output_dir}/")

    # Classifies sentiment as positve or negative.
    def classify_sentiment(self, text):
        # Don't track gradient.
        with torch.no_grad():
            # Tokens are made up of CLS token, text converted to tokens, and SEP token.
            tokens = ["[CLS]"] + self.tokenizer.tokenize(text) + ["[SEP]"]
            # Convert tokens to input IDs; convert them to tensor, unsqueeze, put it to device.
            input_ids = (
                torch.tensor(self.tokenizer.convert_tokens_to_ids(tokens))
                .unsqueeze(0)
                .to(self.device)
            )
            # Create attention mask from input IDs.
            attention_mask = (input_ids != 0).long()
            # Get logit (log-odds) of sentiment being positive from the model.
            positive_logit = self.model(
                input_ids=input_ids, attention_mask=attention_mask
            )
            # Convert the logit to a probability.
            positive_probability = torch.sigmoid(positive_logit.unsqueeze(-1)).item()
            # Convert the probability to a percentage.
            positive_percentage = positive_probability * 100
            # Conver probability to boolean.
            is_positive = positive_probability > 0.5
            # Return sentiment and percentage.
            if is_positive:
                return "Positive", int(positive_percentage)
            else:
                return "Negative", int(100 - positive_percentage)
