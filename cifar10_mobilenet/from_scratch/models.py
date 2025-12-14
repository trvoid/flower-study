import timm
import torch

def get_model(model_name, pretrained, num_classes, device):
    """timm을 사용하여 모델 로드"""

    # Map user-friendly names to timm model names
    model_mapping = {
        'efficientnet': 'efficientnet_b0',
        'mobilenet': 'mobilenetv3_small_100',
        'wideresnet': 'wide_resnet28_10',
        'vit': 'vit_base_patch16_224',
    }
    
    if model_name not in model_mapping:
        raise ValueError(f"Unknown model: {model_name}. Available: {list(model_mapping.keys())}")
        
    timm_name = model_mapping[model_name]

    model = timm.create_model(
        timm_name, 
        pretrained=pretrained,
        num_classes=num_classes
    )

    return model.to(device)
