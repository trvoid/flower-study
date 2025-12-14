import timm

def get_model(model_name, pretrained=True, num_classes=10, img_size=32, device='cuda'):
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

    # Adjust stride for small images (CIFAR-10 32x32)
    if img_size <= 32:
        print(f"Adjusting stride for small image size: {img_size}")
        if hasattr(model, 'conv_stem'):
            # EfficientNet, MobileNet
            model.conv_stem.stride = (1, 1)
        elif hasattr(model, 'conv1'):
            # ResNet
            model.conv1.stride = (1, 1)

    return model.to(device)
