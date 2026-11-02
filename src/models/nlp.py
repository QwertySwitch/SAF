import torch
import torch.nn as nn
from transformers import AutoModelForSequenceClassification, AutoTokenizer, AutoConfig
from typing import Dict, Optional, Tuple

class RoBERTaModel(nn.Module):

    def __init__(self, num_labels, model_name='FacebookAI/roberta-base', pretrained=True, dropout=0.1, gradient_checkpointing=False):
        super().__init__()
        self.model_name = model_name
        self.num_labels = num_labels
        self.config = AutoConfig.from_pretrained(model_name, num_labels=num_labels, hidden_dropout_prob=dropout, attention_probs_dropout_prob=dropout)
        if pretrained:
            self.model = AutoModelForSequenceClassification.from_pretrained(model_name, config=self.config)
        else:
            self.model = AutoModelForSequenceClassification.from_config(self.config)
        if gradient_checkpointing and hasattr(self.model, 'gradient_checkpointing_enable'):
            self.model.gradient_checkpointing_enable()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

    def forward(self, input_ids, attention_mask=None, token_type_ids=None, labels=None, **kwargs):
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids, labels=labels, **kwargs)
        return {'loss': outputs.loss if labels is not None else None, 'logits': outputs.logits, 'hidden_states': outputs.hidden_states if hasattr(outputs, 'hidden_states') else None, 'attentions': outputs.attentions if hasattr(outputs, 'attentions') else None}

    def get_trainable_params(self):
        return {name: param for (name, param) in self.model.named_parameters() if param.requires_grad}

    def freeze_backbone(self):
        for (name, param) in self.model.named_parameters():
            if 'classifier' not in name:
                param.requires_grad = False

    def unfreeze_all(self):
        for param in self.model.parameters():
            param.requires_grad = True

    def get_num_params(self):
        total_params = sum((p.numel() for p in self.model.parameters()))
        trainable_params = sum((p.numel() for p in self.model.parameters() if p.requires_grad))
        return (total_params, trainable_params)

    def save_pretrained(self, save_directory):
        self.model.save_pretrained(save_directory)
        self.tokenizer.save_pretrained(save_directory)

    @classmethod
    def from_pretrained(cls, load_directory, num_labels):
        instance = cls(num_labels=num_labels, pretrained=False)
        instance.model = AutoModelForSequenceClassification.from_pretrained(load_directory)
        instance.tokenizer = AutoTokenizer.from_pretrained(load_directory)
        return instance
RoBERTaSmallModel = RoBERTaModel

def get_nlp_model(task, num_labels, model_name='FacebookAI/roberta-base', pretrained=True, dropout=0.1, gradient_checkpointing=False):
    return RoBERTaModel(num_labels=num_labels, model_name=model_name, pretrained=pretrained, dropout=dropout, gradient_checkpointing=gradient_checkpointing)
