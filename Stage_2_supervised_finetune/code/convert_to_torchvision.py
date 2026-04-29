#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Convert DINO backbone weights to torchvision.models.regnet_y_800mf format
Remove both 'module.' and 'backbone.' prefixes
"""
import torch
import os

checkpoint_path = '../../dino/models/checkpoint.pth'
# checkpoint_path = '../checkpoint/checkpoint.pth'

output_dir = '../checkpoint'

save_path = os.path.join(output_dir, 'teacher_backbone_torchvision.pth')

print("Loading checkpoint...")
checkpoint = torch.load(checkpoint_path, map_location='cpu')

# Get teacher backbone weights
teacher = checkpoint['teacher']

# Filter only backbone keys
backbone_keys = [k for k in teacher.keys() if 'backbone' in k]
print("Total backbone keys in checkpoint: {}".format(len(backbone_keys)))

# Convert to torchvision format (remove 'module.' and 'backbone.' prefixes)
torchvision_state_dict = {}
for old_key in backbone_keys:
    # Remove 'module.' prefix
    new_key = old_key.replace('module.', '', 1) if old_key.startswith('module.') else old_key
    # Remove 'backbone.' prefix
    new_key = new_key.replace('backbone.', '', 1) if new_key.startswith('backbone.') else new_key
    torchvision_state_dict[new_key] = teacher[old_key]

print("\n=== Conversion mapping (first 10) ===")
for old_key in backbone_keys[:10]:
    new_key = old_key.replace('module.', '', 1) if old_key.startswith('module.') else old_key
    new_key = new_key.replace('backbone.', '', 1) if new_key.startswith('backbone.') else new_key
    print("  {} -> {}".format(old_key, new_key))

# Save converted weights
torch.save(torchvision_state_dict, save_path)
print("\nSaved to: {}".format(save_path))
print("File size: {:.2f} MB".format(os.path.getsize(save_path) / (1024*1024)))

# Verify with torchvision
print("\n=== Verifying with torchvision.models.regnet_y_800mf ===")
try:
    from torchvision.models import regnet_y_800mf
    model = regnet_y_800mf(pretrained=False)
    model_keys = set(model.state_dict().keys())
    converted_keys = set(torchvision_state_dict.keys())
    
    # Check key coverage
    matching = model_keys & converted_keys
    missing_in_converted = model_keys - converted_keys
    extra_in_converted = converted_keys - model_keys
    
    print("Matching keys: {}".format(len(matching)))
    print("Missing in converted: {}".format(len(missing_in_converted)))
    print("Extra in converted: {}".format(len(extra_in_converted)))
    
    if missing_in_converted:
        print("\nMissing keys (in torchvision but not in converted):")
        for k in sorted(missing_in_converted)[:20]:
            print("  {}".format(k))
        if len(missing_in_converted) > 20:
            print("  ... and {} more".format(len(missing_in_converted) - 20))
    
    if extra_in_converted:
        print("\nExtra keys (in converted but not in torchvision):")
        for k in sorted(extra_in_converted)[:20]:
            print("  {}".format(k))
        if len(extra_in_converted) > 20:
            print("  ... and {} more".format(len(extra_in_converted) - 20))
    
    # Check shape matching
    print("\n=== Shape verification ===")
    shape_mismatch = []
    for k in matching:
        model_shape = model.state_dict()[k].shape
        converted_shape = torchvision_state_dict[k].shape
        if model_shape != converted_shape:
            shape_mismatch.append((k, model_shape, converted_shape))
    
    if shape_mismatch:
        print("Shape mismatches:")
        for k, ms, cs in shape_mismatch[:10]:
            print("  {}: model={}, converted={}".format(k, ms, cs))
        if len(shape_mismatch) > 10:
            print("  ... and {} more mismatches".format(len(shape_mismatch) - 10))
    else:
        print("All shapes match!")
    
    # Test loading
    print("\n=== Test load_state_dict ===")
    result = model.load_state_dict(torchvision_state_dict, strict=False)
    print("Missing keys: {}".format(result.missing_keys))
    print("Unexpected keys: {}".format(result.unexpected_keys))
    
    if not result.missing_keys and not result.unexpected_keys:
        print("\n*** SUCCESS: All weights loaded correctly! ***")
    elif not result.unexpected_keys:
        print("\n*** SUCCESS: All backbone weights loaded! (fc layer missing is expected) ***")
    
except ImportError as e:
    print("torchvision not available: {}".format(e))

print("\n" + "="*80)
print("Conversion complete!")