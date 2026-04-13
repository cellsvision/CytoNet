#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Extract model weights from DINO checkpoint and save separately
"""
import torch
import os

checkpoint_path = '../checkpoint/checkpoint.pth'
output_dir = '../checkpoint'

print("Loading checkpoint...")
checkpoint = torch.load(checkpoint_path, map_location='cpu')

# 1. Extract student backbone weights (remove 'module.' prefix)
print("\n=== Extracting student backbone weights ===")
student_backbone = {}
for k, v in checkpoint['student'].items():
    if 'backbone' in k:
        # Remove 'module.' prefix
        new_key = k.replace('module.', '', 1) if k.startswith('module.') else k
        student_backbone[new_key] = v

print("Student backbone keys: {}".format(len(student_backbone)))
save_path = os.path.join(output_dir, 'student_backbone.pth')
torch.save(student_backbone, save_path)
print("Saved to: {}".format(save_path))
print("File size: {:.2f} MB".format(os.path.getsize(save_path) / (1024*1024)))

# 2. Extract teacher backbone weights (remove 'module.' prefix)
print("\n=== Extracting teacher backbone weights ===")
teacher_backbone = {}
for k, v in checkpoint['teacher'].items():
    if 'backbone' in k:
        new_key = k.replace('module.', '', 1) if k.startswith('module.') else k
        teacher_backbone[new_key] = v

print("Teacher backbone keys: {}".format(len(teacher_backbone)))
save_path = os.path.join(output_dir, 'teacher_backbone.pth')
torch.save(teacher_backbone, save_path)
print("Saved to: {}".format(save_path))
print("File size: {:.2f} MB".format(os.path.getsize(save_path) / (1024*1024)))

# 3. Extract full student model (backbone + head, remove 'module.' prefix)
print("\n=== Extracting full student model ===")
student_full = {}
for k, v in checkpoint['student'].items():
    new_key = k.replace('module.', '', 1) if k.startswith('module.') else k
    student_full[new_key] = v

print("Student full model keys: {}".format(len(student_full)))
save_path = os.path.join(output_dir, 'student_full.pth')
torch.save(student_full, save_path)
print("Saved to: {}".format(save_path))
print("File size: {:.2f} MB".format(os.path.getsize(save_path) / (1024*1024)))

# 4. Extract full teacher model (backbone + head, remove 'module.' prefix)
print("\n=== Extracting full teacher model ===")
teacher_full = {}
for k, v in checkpoint['teacher'].items():
    new_key = k.replace('module.', '', 1) if k.startswith('module.') else k
    teacher_full[new_key] = v

print("Teacher full model keys: {}".format(len(teacher_full)))
save_path = os.path.join(output_dir, 'teacher_full.pth')
torch.save(teacher_full, save_path)
print("Saved to: {}".format(save_path))
print("File size: {:.2f} MB".format(os.path.getsize(save_path) / (1024*1024)))

# 5. Extract student backbone with original 'module.' prefix (for DDP loading)
print("\n=== Extracting student backbone (DDP format) ===")
student_backbone_ddp = {}
for k, v in checkpoint['student'].items():
    if 'backbone' in k:
        student_backbone_ddp[k] = v

print("Student backbone DDP keys: {}".format(len(student_backbone_ddp)))
save_path = os.path.join(output_dir, 'student_backbone_ddp.pth')
torch.save(student_backbone_ddp, save_path)
print("Saved to: {}".format(save_path))
print("File size: {:.2f} MB".format(os.path.getsize(save_path) / (1024*1024)))

# Summary
print("\n" + "="*80)
print("=== Extraction Summary ===")
print("="*80)
print("\nGenerated files:")
print("  1. student_backbone.pth    - Student backbone only (no 'module.' prefix)")
print("  2. teacher_backbone.pth    - Teacher backbone only (no 'module.' prefix)")
print("  3. student_full.pth        - Full student model (backbone + head)")
print("  4. teacher_full.pth        - Full teacher model (backbone + head)")
print("  5. student_backbone_ddp.pth - Student backbone with 'module.' prefix")
print("\nNote: Teacher model is typically better for downstream tasks (EMA updated)")
print("Extraction complete!")