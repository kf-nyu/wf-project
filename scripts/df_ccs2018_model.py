"""
PyTorch port of the closed-world *non-defended* DF model from the official repo:

  deep-fingerprinting/df  →  src/Model_NoDef.py  (Keras)  →  DFNet.build

Reference (paper + code):
  Sirinam et al., "Deep Fingerprinting", ACM CCS 2018.
  https://github.com/deep-fingerprinting/df/blob/master/src/Model_NoDef.py

Differences from Keras (document for the paper):
- Last layer is **logits** (no softmax); train with :class:`torch.nn.CrossEntropyLoss`.
- Max-pooling uses ``padding=0`` (Keras used ``padding='same'``); the inferred ``flat_dim``
  matches the length produced by this stack for ``seq_len=5000`` so FC sizes align with the
  original parameter *layout* intent; a tiny length mismatch vs. TensorFlow is possible and
  should be described as a porting detail if you need bit-identical activations.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class DFNetCCS2018(nn.Module):
    """
    4 convolutional blocks (2 conv1d each), two 512-FCs with BN/ReLU/dropout, then classifier.

    Input: ``(batch, 1, seq_len)`` (direction sequence), same as our DF95 pipeline.
    Output: class **logits** of shape ``(batch, num_classes)``.
    """

    def __init__(self, num_classes: int, seq_len: int = 5000) -> None:
        super().__init__()
        # Mirrors Model_NoDef.DFNet — filter_num [None, 32, 64, 128, 256], kernel 8, pool 8 stride 4
        self.features = nn.Sequential(
            # Block 1: ELU in paper code
            nn.Conv1d(1, 32, kernel_size=8, stride=1, padding=4),
            nn.BatchNorm1d(32),
            nn.ELU(alpha=1.0, inplace=True),
            nn.Conv1d(32, 32, kernel_size=8, stride=1, padding=4),
            nn.BatchNorm1d(32),
            nn.ELU(alpha=1.0, inplace=True),
            nn.MaxPool1d(kernel_size=8, stride=4, padding=0),
            nn.Dropout(0.1),
            # Block 2
            nn.Conv1d(32, 64, kernel_size=8, stride=1, padding=4),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.Conv1d(64, 64, kernel_size=8, stride=1, padding=4),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=8, stride=4, padding=0),
            nn.Dropout(0.1),
            # Block 3
            nn.Conv1d(64, 128, kernel_size=8, stride=1, padding=4),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Conv1d(128, 128, kernel_size=8, stride=1, padding=4),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=8, stride=4, padding=0),
            nn.Dropout(0.1),
            # Block 4
            nn.Conv1d(128, 256, kernel_size=8, stride=1, padding=4),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Conv1d(256, 256, kernel_size=8, stride=1, padding=4),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=8, stride=4, padding=0),
            nn.Dropout(0.1),
        )
        with torch.no_grad():
            z = self.features(torch.zeros(1, 1, seq_len))
            flat_dim = z.numel()
        self.fc1 = nn.Linear(flat_dim, 512)
        self.bn_fc1 = nn.BatchNorm1d(512)
        self.fc2 = nn.Linear(512, 512)
        self.bn_fc2 = nn.BatchNorm1d(512)
        self.drop_fc1 = nn.Dropout(0.7)
        self.drop_fc2 = nn.Dropout(0.5)
        self.head = nn.Linear(512, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.features(x).flatten(1)
        h = self.drop_fc1(F.relu(self.bn_fc1(self.fc1(h)), inplace=True))
        h = self.drop_fc2(F.relu(self.bn_fc2(self.fc2(h)), inplace=True))
        return self.head(h)
