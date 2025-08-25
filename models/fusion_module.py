import torch
import pytorch_lightning as pl
from typing import List

from dataclasses import dataclass
import torch.nn as nn
import numpy as np

# import our library
# from sklearn.metrics import f1_score, top_k_accuracy_score
from sklearn.metrics import f1_score as sklearn_f1_score
from sklearn.metrics import roc_auc_score
from torchmetrics.functional import f1_score
from torchmetrics.functional.classification import accuracy
# initialize metric
from sklearn.metrics import classification_report

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


class LitMMFusionModel(pl.LightningModule):

    def __init__(self, model, classes, criterion, lr, task="multiclass"):
        super(LitMMFusionModel, self).__init__()
        self.save_hyperparameters("classes", "lr", "task")
        self.model = model
        self.criterion = criterion
        self.lr = lr
        self.classes = [str(c) for c in classes]
        self.task = task
        self.val_preds = []
        self.val_true = []
        self.test_preds = []
        self.test_true = []
        print("task is ", self.task)

    def forward(self, x, *args, **kwargs):
        return self.model(x)

    def top_k_accuracy(self, y_true, y_pred, k):
        # Get top-k predictions along the class dimension
        top_k_preds = torch.topk(y_pred, k=k, dim=1).indices
        # Check if the true class is among the top-k predictions
        matches = torch.any(top_k_preds == y_true.view(-1, 1), dim=1)
        # Compute accuracy
        accuracy = matches.float().mean().item()
        return accuracy

    def __evaluation_metrics(self, out_logits, y_true, loss, split="train"):

        labels = list(range(len(self.classes)))
        logs = {}
        # Compute probabilities and labels
        num_classes = len(self.classes)
        y_probs = None
        if self.task == "multilabel":
            y_probs = torch.sigmoid(out_logits)
            y_pred_labels = y_probs.round()
        elif self.task == "binary":
            y_probs = torch.sigmoid(out_logits)
            y_pred_labels = y_probs.round()
        elif self.task == "multiclass":
            y_probs = torch.softmax(out_logits, dim=-1)
            y_pred_labels = y_probs.argmax(dim=-1)
            # For multiclass, ensure y_true is 1D and y_probs is 2D

        if split == "val":
            self.val_true.extend(y_true.tolist())
            self.val_preds.extend(y_pred_labels.tolist())
        elif split == "test":
            self.test_true.extend(y_true.tolist())
            self.test_preds.extend(y_pred_labels.tolist())
        # acc = self.top_k_accuracy(y_true, y_pred, k=1)
        acc = accuracy(y_probs, y_true, task=self.task,
                       num_classes=num_classes)
        if len(self.classes) >= 6:
            top3 = self.top_k_accuracy(y_true, y_probs, k=3)
            top5 = self.top_k_accuracy(y_true, y_probs, k=5)

           # top3 = accuracy(y_probs,y_true,task=self.task,num_classes=num_classes,top_k=3)
           # top5 = accuracy(y_probs,y_true,task=self.task,num_classes=num_classes,top_k=5)
            logs.update({f'{split}_top3': top3, f'{split}_top5': top5})

        # Replace torchmetrics auroc with sklearn roc_auc_score
        auroc_value = 0
        # Convert tensors to numpy for sklearn
        y_true_np = y_true.detach().cpu().numpy()
        y_probs_np = y_probs.detach().cpu().numpy()
        try:
            if self.task == "binary":
                # For binary classification, use probabilities of positive class
                auroc_value = roc_auc_score(y_true_np, y_probs_np)
            elif self.task == "multiclass":
                if y_probs_np.ndim == 2 and y_probs_np.shape[1] == 2:
                    y_probs_np = y_probs_np[:, 1]
                    auroc_value = roc_auc_score(y_true_np, y_probs_np)
                else:
                    auroc_value = roc_auc_score(y_true_np, y_probs_np,
                                                multi_class='ovr', average='macro', labels=labels)
            elif self.task == "multilabel":
                # For multilabel, compute average across all labels
                auroc_value = roc_auc_score(
                    y_true_np, y_probs_np, average='macro')
        except ValueError as e:
            print("y true", y_true_np)
            print("y probs", y_probs_np)
            print("auroc value", auroc_value)
            print("error", e)
            raise e

        # Add sklearn f1 scores

        # Convert tensors to numpy for sklearn
        y_true_np = y_true.detach().cpu().numpy()
        y_pred_np = y_pred_labels.detach().cpu().numpy()

        sklearn_f1_macro = sklearn_f1_score(
            y_true_np, y_pred_np, average="macro")
        sklearn_f1_micro = sklearn_f1_score(
            y_true_np, y_pred_np, average="micro")
        sklearn_f1_weighted = sklearn_f1_score(
            y_true_np, y_pred_np, average="weighted")

        logs.update({f'{split}_loss': loss, f'{split}_acc': acc, f'{split}_auroc': auroc_value,
                     f'sklearn_{split}_f1_macro': sklearn_f1_macro, f'sklearn_{split}_f1_micro': sklearn_f1_micro,
                     f'sklearn_{split}_f1_weighted': sklearn_f1_weighted})

        return logs

    def training_step(self, batch, batch_idx):

        target = batch['label']
        data_input = {k: v for k, v in batch.items() if k != 'label'}
        target = target.float() if self.task != "multiclass" else target.long()

        output = self.model(data_input)
        loss = self.criterion(output, target)

        logs = self.__evaluation_metrics(output, target, loss, split="train")
        self.log_dict(logs, on_step=False, on_epoch=True,
                      prog_bar=True, logger=True, batch_size=target.shape[0])
        return loss

    def validation_step(self, batch, batch_idx):
        target = batch['label']
        data_input = {k: v for k, v in batch.items() if k != 'label'}
        target = target.float() if self.task != "multiclass" else target.long()
        output = self.model(data_input)
        loss = self.criterion(output, target)
        logs = self.__evaluation_metrics(output, target, loss, split="val")
        self.log_dict(logs, on_step=False, on_epoch=True,
                      prog_bar=True, logger=True, batch_size=target.shape[0])
        return loss

   # def on_fit_start(self):
       # self.model.encoders["text"].features_extractor=self.model.encoders["text"].features_extractor.cpu()
        # Print device information for all encoders in the model
       # torch.cuda.empty_cache()
    def on_validation_end(self):
        from sklearn.metrics import classification_report

        # Convert tensors to numpy for sklearn
        y_true_np = np.array(self.val_true)
        y_pred_np = np.array(self.val_preds)
        # Get unique classes present in validation data

        # Generate classification report
        print("Classification Report validation set:")
        print(classification_report(y_true_np, y_pred_np))
        self.val_true.clear()
        self.val_preds.clear()

    def on_test_end(self):
        from sklearn.metrics import classification_report, confusion_matrix

        # Convert tensors to numpy for sklearn
        y_true_np = np.array(self.test_true)
        y_pred_np = np.array(self.test_preds)

        # Generate classification report
        print("Classification Report test set:")
        print(classification_report(y_true_np, y_pred_np))
        print("Confusion Matrix test set:")
        print(confusion_matrix(y_true_np, y_pred_np))
        self.test_true.clear()
        self.test_preds.clear()

    def test_step(self, batch, batch_idx):

        target = batch['label']
        data_input = {k: v for k, v in batch.items() if k != 'label'}
        target = target.float() if self.task != "multiclass" else target.long()
        output = self.model(data_input)
        loss = self.criterion(output, target)
        logs = self.__evaluation_metrics(output, target, loss, split="test")
        self.log_dict(
            logs,
            on_step=False, on_epoch=True, prog_bar=True, logger=True, batch_size=target.shape[0]
        )
        return torch.sigmoid(output).round().long() if self.task in ["multilabel", "binary"] else output.argmax(1)

    def configure_optimizers(self):
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        return {"optimizer": self.optimizer}


class UnimodalModel(nn.Module):
    def __init__(self, encoders: dict, classifier) -> None:
        super(UnimodalModel, self).__init__()
        self.encoders = nn.ModuleDict(encoders)
        self.classifier = classifier

    def forward(self, x):
        only_key = next(iter(self.encoders.keys()))
        encoded = self.encoders[only_key](x[only_key])
        out = self.classifier(encoded)
        return out


class FusionModel(nn.Module):
    def __init__(self, encoders: dict, fusion, classifier) -> None:
        super(FusionModel, self).__init__()
        self.encoders = nn.ModuleDict(encoders)
        self.classifier = classifier
        self.fusion = fusion

    def forward(self, x):
        # Initialize a list to store encoded outputs
        encoded_outputs = []

        # Process image data if available
        if "image" in x and "image" in self.encoders:
            image_output = self.encoders["image"](x["image"])
            encoded_outputs.append(image_output)

        # Process tabular data if available
        if "tabular" in x and "tabular" in self.encoders:
            tabular_output = self.encoders["tabular"](x["tabular"])
            encoded_outputs.append(tabular_output)

        # Apply fusion if available
        if self.fusion is not None:
            fused_output = self.fusion(encoded_outputs)
            out = self.classifier(fused_output)
        elif len(encoded_outputs) == 1:
            out = self.classifier(encoded_outputs[0])
        else:
            raise ValueError(
                "Fusion is required when multiple modalities are present.")

        return out
