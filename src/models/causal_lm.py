import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
from typing import Dict, Optional, Tuple

class GPT2CausalLMModel(nn.Module):

    def __init__(self, model_name='gpt2', pretrained=True, gradient_checkpointing=False):
        super().__init__()
        self.model_name = model_name
        if pretrained:
            self.model = AutoModelForCausalLM.from_pretrained(model_name)
        else:
            config = AutoConfig.from_pretrained(model_name)
            self.model = AutoModelForCausalLM.from_config(config)
        if gradient_checkpointing and hasattr(self.model, 'gradient_checkpointing_enable'):
            self.model.gradient_checkpointing_enable()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.model.config.pad_token_id = self.tokenizer.eos_token_id

    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.model, name)

    def forward(self, input_ids, attention_mask=None, labels=None, **kwargs):
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        return {'loss': outputs.loss if labels is not None else None, 'logits': outputs.logits, 'hidden_states': outputs.hidden_states if hasattr(outputs, 'hidden_states') else None, 'attentions': outputs.attentions if hasattr(outputs, 'attentions') else None}

    def get_trainable_params(self):
        return {name: param for (name, param) in self.model.named_parameters() if param.requires_grad}

    def freeze_backbone(self):
        for (name, param) in self.model.named_parameters():
            if 'lm_head' not in name:
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
    def from_pretrained(cls, load_directory):
        instance = cls(pretrained=False)
        instance.model = AutoModelForCausalLM.from_pretrained(load_directory)
        instance.tokenizer = AutoTokenizer.from_pretrained(load_directory)
        return instance

def get_causal_lm_model(model_name='gpt2', pretrained=True, gradient_checkpointing=False):
    return GPT2CausalLMModel(model_name=model_name, pretrained=pretrained, gradient_checkpointing=gradient_checkpointing)
