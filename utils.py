import os
import wandb
from typing import Dict,List
from torch import nn
import torch
from transformers import PreTrainedTokenizer
import evaluate

def create_dirs_if_not_exist(path: str) -> None:
    """
    Create directories if they do not exist.

    Args:
        path (str): The path to the directory that needs to be created.

    Returns:
        None
    """
    is_exist = os.path.exists(path)

    if not is_exist:
        os.makedirs(path)
        print("The new directory is created!")


def set_weight_decay(model: nn.Module, weight_decay: float = 0.01, no_decay: List[str] = ["bias", "LayerNorm.weight"]) -> List[dict]:
    """
    Set weight decay for different parameter groups in the optimizer.

    Args:
        model (nn.Module): The neural network model.
        weight_decay (float, optional): The weight decay value. Defaults to 0.01.
        no_decay (List[str], optional): List of parameter names that should not have weight decay. Defaults to ["bias", "LayerNorm.weight"].

    Returns:
        List[dict]: A list of dictionaries specifying parameters and their respective weight decay.
    """
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]
    return optimizer_grouped_parameters

def define_metrics(metric_info: Dict[str, str]) -> None:
    """
    Define custom metrics for tracking in Weights & Biases.

    Args:
        metric_info (Dict[str, str]): A dictionary where keys are metric names and values are step metrics.
        
    Returns:
        None
    """
    for metric_name, step_metric in metric_info.items():
        wandb.define_metric(metric_name, step_metric=step_metric)

def compute_metrics(out: List[torch.Tensor], labels: List[torch.Tensor], tokenizer: PreTrainedTokenizer) -> tuple:
    """
    Compute Character Error Rate (CER) and Word Error Rate (WER) metrics.

    Args:
        out (List[torch.Tensor]): List of output tensors from the model.
        labels (List[torch.Tensor]): List of label tensors.
        tokenizer (PreTrainedTokenizer): Tokenizer for decoding tensors.

    Returns:
        tuple: A tuple containing the computed CER and WER.
    """

    metrics_wer = evaluate.load("wer")
    metrics_cer = evaluate.load("cer")

    o_list, l_list = [], []

    for o, l in zip(out, labels):
        o = torch.argmax(o, dim=1)
        o_list.append(tokenizer.decode(o, skip_special_tokens=True))
        l_list.append(tokenizer.decode(l, skip_special_tokens=True))

    cer = metrics_cer(references=l_list, predictions=o_list)
    wer = metrics_wer(references=l_list, predictions=o_list)

    return cer, wer