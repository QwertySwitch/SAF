import torch
import torch.nn as nn
import timm
from typing import Dict, Optional, Tuple

class _PeftCompatConfig:
    use_return_dict = True
    tie_word_embeddings = False

    def get(self, key, default=None):
        return getattr(self, key, default)

class ViTSmallModel(nn.Module):

    def __init__(self, num_classes, model_name='vit_small_patch16_224.augreg_in21k', pretrained=True, image_size=224, dropout=0.1, gradient_checkpointing=False):
        super().__init__()
        self.model_name = model_name
        self.num_classes = num_classes
        self.image_size = image_size
        self.config = _PeftCompatConfig()
        self.model = timm.create_model(model_name, pretrained=pretrained, num_classes=num_classes, drop_rate=dropout, img_size=image_size)
        if gradient_checkpointing and hasattr(self.model, 'set_grad_checkpointing'):
            self.model.set_grad_checkpointing(enable=True)

    def forward(self, images, labels=None, **kwargs):
        logits = self.model(images)
        loss = None
        if labels is not None:
            loss_fn = nn.CrossEntropyLoss()
            loss = loss_fn(logits, labels)
        return {'loss': loss, 'logits': logits}

    def get_trainable_params(self):
        return {name: param for (name, param) in self.model.named_parameters() if param.requires_grad}

    def freeze_backbone(self):
        for (name, param) in self.model.named_parameters():
            if 'head' not in name:
                param.requires_grad = False

    def unfreeze_all(self):
        for param in self.model.parameters():
            param.requires_grad = True

    def get_num_params(self):
        total_params = sum((p.numel() for p in self.model.parameters()))
        trainable_params = sum((p.numel() for p in self.model.parameters() if p.requires_grad))
        return (total_params, trainable_params)

    def save_checkpoint(self, filepath):
        torch.save({'model_name': self.model_name, 'num_classes': self.num_classes, 'image_size': self.image_size, 'state_dict': self.model.state_dict()}, filepath)

    @classmethod
    def from_checkpoint(cls, filepath):
        checkpoint = torch.load(filepath, map_location='cpu', weights_only=False)
        instance = cls(num_classes=checkpoint['num_classes'], model_name=checkpoint['model_name'], pretrained=False, image_size=checkpoint.get('image_size', 224))
        instance.model.load_state_dict(checkpoint['state_dict'])
        return instance

def get_vision_model(task, num_classes, model_name='vit_small_patch16_224.augreg_in21k', pretrained=True, image_size=224, dropout=0.1, gradient_checkpointing=False):
    return ViTSmallModel(num_classes=num_classes, model_name=model_name, pretrained=pretrained, image_size=image_size, dropout=dropout, gradient_checkpointing=gradient_checkpointing)
