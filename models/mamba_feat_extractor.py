import torch
import numpy as np
import torch.nn as nn
from models.hr_mamba import HRMambaRegressor
from torch.utils.data import DataLoader, TensorDataset


class HRMambaFeatureExtractor(nn.Module):
    
    def __init__(self, model, num_blocks=None, use_norm=False, pool="mean"):
        super().__init__()

        self.patch_embed = model.patch_embed

        if num_blocks is None:
            num_blocks = len(model.blocks)

        self.blocks = nn.ModuleList(model.blocks[:num_blocks])
        self.norm = model.norm if use_norm else nn.Identity()
        self.pool = pool

    def forward(self, x):
        """
        Accepts:
            [B, 3, T]  -> your current case
        or:
            [B, T, 3]

        Returns:
            [B, 128] if pool is "mean" or "last"
        """

        # If input is [B, T, 3], convert to [B, 3, T]
        if x.shape[-1] == 3 and x.shape[1] != 3:
            x = x.transpose(1, 2)

        # If input is already [B, 3, T], keep it
        elif x.shape[1] == 3:
            pass

        else:
            raise ValueError(f"Unexpected input shape: {x.shape}")

        # [B, 3, T] -> [B, 128, T_patches]
        x = self.patch_embed(x)

        # [B, 128, T_patches] -> [B, T_patches, 128]
        x = x.transpose(1, 2)

        for block in self.blocks:
            x = block(x)

        # optional final norm
        x = self.norm(x)

        if self.pool == "mean":
            x = x.mean(dim=1)

        elif self.pool == "last":
            x = x[:, -1, :]

        elif self.pool is None:
            pass

        else:
            raise ValueError("pool must be 'mean', 'last', or None")

        return x

def make_feats(X: np.array, model: HRMambaRegressor) -> torch.tensor:

    """ X.shape = 3D array = [lines, columns, window size] """

    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    # Keep X_train on CPU
    X = torch.tensor(X, dtype=torch.float32)  # shape ex: [86153, 3, 300]

    feature_extractor = HRMambaFeatureExtractor(
        model,
        num_blocks=4,
        use_norm=False,   # stop before LayerNorm
        pool="mean"
    )

    feature_extractor = feature_extractor.to(device)
    feature_extractor.eval()

    loader = DataLoader(
        TensorDataset(X),
        batch_size=512,   # reduce to 256 or 128 if OOM
        shuffle=False
    )

    all_features = []

    with torch.inference_mode():
        for (x_batch,) in loader:
            x_batch = x_batch.to(device)

            feats = feature_extractor(x_batch)
            # feats shape: [batch_size, 128]

            all_features.append(feats.cpu())

    features = torch.cat(all_features, dim=0)

    print("Total feats extracted: ", features.shape)

    return features