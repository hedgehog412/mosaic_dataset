import torch
from torch.utils.data import Dataset
from MosaicDataset import MosaicDataset


class MosaicSessionDataset(Dataset):
    """
    A wrapper around MosaicDataset that returns full session sequences.
    
    Instead of returning individual board samples, this wrapper returns all rendered boards
    from a single session stacked as a tensor of shape (T, C, H, W), where T is the number 
    of steps in the session. This is useful for training next-step generation models.
    """
    
    def __init__(self, log_dir: str, copy_num=1, **kwargs):
        """
        Initialize the session dataset wrapper.
        
        Args:
            log_dir: Directory containing session JSONL files
            copy_num: Number of augmented copies per session (default 1)
            **kwargs: Additional arguments passed to MosaicDataset
                (include_intermediate, global_seed, aug_img_p, jitter_px, dropout_p,
                 rot_deg_max, scale_p, chip_p, chip_max_frac, edge_dropout_p,
                 enable_augmentation, crop_padding_frac, crop_jitter_frac, output_size, min_tiles)
        """
        # Create underlying MosaicDataset with include_intermediate=True to get all steps
        self.mosaic_dataset = MosaicDataset(
            log_dir=log_dir,
            include_intermediate=True,
            copy_num=1,  # We handle multiple copies at wrapper level
            **kwargs
        )
        
        self.copy_num = copy_num
        self.min_tiles = kwargs.get('min_tiles', 1)
        
        # Build mapping from session index to the indices in base_idx for that session
        self.session_indices = []
        for si, session in enumerate(self.mosaic_dataset.sessions):
            session_base_indices = [i for i, (s_idx, _) in enumerate(self.mosaic_dataset.base_idx) if s_idx == si]
            if len(session_base_indices) > 0:
                self.session_indices.append(session_base_indices)
    
    def set_epoch(self, epoch: int):
        """Set the epoch for epoch-dependent randomness."""
        self.mosaic_dataset.set_epoch(epoch)
    
    def __len__(self):
        """Return the number of (session, augmentation copy) pairs."""
        return len(self.session_indices) * self.copy_num
    
    def __getitem__(self, idx):
        """
        Get all rendered boards from a session as a stacked tensor.
        
        Args:
            idx: Index into the dataset
            
        Returns:
            Tensor of shape (T, C, H, W) where T is the number of steps,
            C is the number of channels (1), H and W are output_size.
        """
        # Map to session and augmentation copy
        base_i = idx // self.copy_num
        aug_i = idx % self.copy_num
        
        # Get the base_idx indices for this session
        session_base_indices = self.session_indices[base_i]
        
        # Calculate seed once for this session/augmentation copy
        # All boards in this session will use the same seed
        si = self.mosaic_dataset.base_idx[session_base_indices[0]][0]  # session index
        seed = (self.mosaic_dataset.global_seed + 100000 * self.mosaic_dataset.epoch + 1000 * si + aug_i) % (2 ** 32 - 1)
        
        # Set fixed seed on the underlying dataset
        self.mosaic_dataset.set_fixed_seed(seed)
        
        try:
            # Render all boards in this session using the underlying dataset
            boards = []
            for base_idx in session_base_indices:
                # Get board from MosaicDataset (returns CHW format)
                # We use base_idx directly with copy_num=1 on the mosaic_dataset
                mosaic_idx = base_idx * self.mosaic_dataset.copy_num
                board = self.mosaic_dataset[mosaic_idx]
                boards.append(board)
            
            # Stack all boards into a single tensor (T, C, H, W)
            stacked_boards = torch.stack(boards, dim=0)
            return stacked_boards
        finally:
            # Always clear the fixed seed to resume normal operation
            self.mosaic_dataset.clear_fixed_seed()
