import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from scipy.ndimage import label

class YoloBackbone(nn.Module):
    def __init__(self, backbone_seq, selected_indices=None):
        super().__init__()
        self.backbone = backbone_seq
        if selected_indices is None:
            # By default, collect all intermediate outputs
            self.selected_indices = list(range(len(backbone_seq)))
        else:
            self.selected_indices = selected_indices

    def forward(self, x):
        feats = []
        out = x
        for i, layer in enumerate(self.backbone):
            # Check if the layer is a Concat layer
            if layer.__class__.__name__.lower().startswith('concat'):
                # If not already a list/tuple, wrap it
                if not isinstance(out, (list, tuple)):
                    out = [out]
            out = layer(out)
            if i in self.selected_indices:
                # If out is a list (as in some YOLO features), add all, else add single
                if isinstance(out, (list, tuple)):
                    feats.extend(out)
                else:
                    feats.append(out)
        return feats


class MultiScaleFusion(nn.Module):
    def __init__(self, in_channels_list, out_channels):
        super().__init__()
        self.reduce_convs = nn.ModuleList([
            nn.Conv2d(in_c, out_channels, 1) for in_c in in_channels_list
        ])
        self.fuse = nn.Sequential(
            nn.Conv2d(len(in_channels_list) * out_channels, out_channels, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.ReLU(inplace=True)
        )

    def forward(self, features):
        target_size = features[0].shape[-2:]
        upsampled = []
        for conv, feat in zip(self.reduce_convs, features):
            out = conv(feat)
            if out.shape[-2:] != target_size:
                out = F.interpolate(out, size=target_size, mode='bilinear', align_corners=False)
            upsampled.append(out)
        fused = torch.cat(upsampled, dim=1)
        return self.fuse(fused)

def soft_argmax(heatmap, beta=100):
    # heatmap: (B, K, H, W)
    *rest, H, W = heatmap.shape
    heatmap_flat = heatmap.reshape(-1, H*W)  # (B*K, H*W)
    y_soft = F.softmax(heatmap_flat * beta, dim=1)

    # Create meshgrid of coordinates
    y_coords, x_coords = torch.meshgrid(
        torch.arange(H, device=heatmap.device),
        torch.arange(W, device=heatmap.device),
        indexing='ij'
    )  # each shape (H, W)
    x_coords = x_coords.reshape(-1).float()  # (H*W,)
    y_coords = y_coords.reshape(-1).float()  # (H*W,)

    x = (y_soft * x_coords).sum(dim=1)
    y = (y_soft * y_coords).sum(dim=1)
    coords = torch.stack((x, y), dim=1)  # (B*K, 2)
    return coords.view(*rest, 2)

def spatial_softmax(heatmap):
    """
    Applies softmax over the spatial dimension of a 4D tensor (B, 1, H, W).
    Returns same shape, values sum to 1 per image.
    """
    B, C, H, W = heatmap.shape
    heatmap_flat = heatmap.view(B, -1)
    softmax_flat = F.softmax(heatmap_flat, dim=1)
    return softmax_flat.view(B, 1, H, W)
    # return heatmap

def thresholded_locations(data_2d, threshold):
    """
    Find centroids of connected components above a threshold.
    Args:
        data_2d: 2D numpy array (can be logits or probabilities).
        threshold: threshold for segmentation.
        from_logits: If True, applies sigmoid to convert logits to probabilities.
    Returns:
        List of centroid coordinates (as np arrays).
    """
    # If input is logits, convert to probability first!
    probs = data_2d

    # Create a binary mask
    thresholded_2d = (probs >= threshold).astype(np.uint8)

    structure = np.ones((3, 3), dtype=int)  # 8-connectivity
    labeled, num_features = label(thresholded_2d, structure=structure)
    centroids = []
    for mesh_label in range(1, num_features + 1):
        positions = np.argwhere(labeled == mesh_label)
        centroid = positions.mean(axis=0)
        centroids.append(centroid)
    return centroids

def combine_nearby_peaks(peaks, distance_threshold=10):
    """
    Combine nearby peaks into single keypoints using clustering.
    
    Args:
        peaks: List of peak coordinates [[y1, x1], [y2, x2], ...]
        distance_threshold: Maximum distance to consider peaks as part of same cluster
    
    Returns:
        List of combined peak coordinates
    """
    if not peaks:
        return []
    
    # Convert to numpy array for easier manipulation
    peaks = np.array(peaks)
    
    # If only one peak, return it
    if len(peaks) == 1:
        return peaks.tolist()
    
    # Calculate pairwise distances
    from scipy.spatial.distance import pdist, squareform
    distances = squareform(pdist(peaks))
    
    # Create clusters
    clusters = []
    used = set()
    
    for i in range(len(peaks)):
        if i in used:
            continue
            
        # Start a new cluster
        cluster = [i]
        used.add(i)
        
        # Find all peaks within distance_threshold
        for j in range(i + 1, len(peaks)):
            if j not in used and distances[i, j] <= distance_threshold:
                cluster.append(j)
                used.add(j)
        
        clusters.append(cluster)
    
    # Calculate centroid for each cluster
    combined_peaks = []
    for cluster in clusters:
        cluster_peaks = peaks[cluster]
        centroid = cluster_peaks.mean(axis=0)
        combined_peaks.append(centroid)
    
    return combined_peaks