import torch
from torch.utils.data import Dataset
import json
from pathlib import Path
from DataClasses import Tile, Event, Session
import numpy as np

class MosaicDataset(Dataset):
    def __init__(self, log_dir: str|Path, \
                include_intermediate=True, copy_num=5, \
                global_seed=12, aug_img_p=0.8, \
                jitter_px=3.0, dropout_p=0.02,\
                rot_deg_max=10.0, scale_p=0.05,\
                chip_p=0.10, chip_max_frac=0.30,\
                edge_dropout_p=0.02, enable_augmentation=True,\
                crop_padding_frac=0.3, crop_jitter_frac=0.1,\
                output_size=(256, 256), min_tiles=1):
        self.log_dir = Path(log_dir)
        self.sessions = [self.load_session(path) for path in self.log_dir.glob("*.jsonl")]
        self.total_sessions = len(self.sessions)
        self.total_steps = 0
        for session in self.sessions:
            self.total_steps += session.num_steps

        self.include_intermediate = include_intermediate
        self.copy_num = copy_num

        self.enable_augmentation = enable_augmentation
        self.aug_img_p = aug_img_p
        self.jitter_px = jitter_px
        self.dropout_p = dropout_p
        self.rot_deg_max = rot_deg_max
        self.scale_p = scale_p
        self.chip_p = chip_p
        self.chip_max_frac = chip_max_frac
        self.edge_dropout_p = edge_dropout_p
        
        # Camera-like cropping parameters
        self.crop_padding_frac = crop_padding_frac  # Padding around tiles as fraction of scene size
        self.crop_jitter_frac = crop_jitter_frac  # Random crop offset as fraction of padding (for augmentation)
        self.output_size = output_size
        self.min_tiles = min_tiles  # Minimum number of tiles required for valid sample

        # Converting dataset idx to session, step idx if intermediate steps included
        # Filter out events with fewer than min_tiles
        self.base_idx = []
        if self.include_intermediate:
            for si, s in enumerate(self.sessions):
                for ei in range(s.num_steps):
                    if len(s.events[ei].canvas_state) >= self.min_tiles:
                        self.base_idx.append((si, ei))
        else:
            # For final states only, filter sessions with insufficient tiles
            for si, s in enumerate(self.sessions):
                if s.num_steps > 0 and len(s.events[s.num_steps - 1].canvas_state) >= self.min_tiles:
                    self.base_idx.append((si, s.num_steps - 1))

        self.global_seed = global_seed
        self.epoch = 0

    # NEEDS TO BE CALLED FOR EACH EPOCH FOR EPOCH DEPENDENT RANDOMNESS
    def set_epoch(self, epoch: int):
        self.epoch = epoch

    def __len__(self):
        return len(self.base_idx) * self.copy_num

    def __getitem__(self, idx):
        # For getting session/event
        base_i = idx // self.copy_num
        (si, ei) = self.base_idx[base_i]
        
        session = self.sessions[si]
        event = session.events[ei]
        
        # Sanity check: should never happen after filtering, but just in case
        if len(event.canvas_state) < self.min_tiles:
            raise ValueError(f"Invalid sample at idx {idx}: only {len(event.canvas_state)} tiles, minimum is {self.min_tiles}")

        # For getting augmentation copy
        aug_i = idx % self.copy_num
        seed = (self.global_seed + 100000 * self.epoch + 1000 * si + 10 * ei + aug_i) % (2 ** 32 - 1)
        rng = np.random.default_rng(seed = seed)
        aug_img = False
        if self.enable_augmentation and aug_i < self.copy_num * self.aug_img_p:
            aug_img = True
        img = self.render_tiles(session.events[ei].canvas_state, session.canvas_width, session.canvas_height, session.tile_size, rng, aug_img=aug_img)
        img = img.permute(2, 0, 1)  # HWC to CHW
        return img
    
    def calculate_bounding_box(self, tiles: list[Tile], tile_size: int, canvas_w: int, canvas_h: int, padding_frac: float):
        """Calculate bounding box around all tiles with padding."""
        if len(tiles) == 0:
            # No tiles, return center crop
            half_w = canvas_w // 4
            half_h = canvas_h // 4
            cx, cy = canvas_w // 2, canvas_h // 2
            return max(0, cx - half_w), max(0, cy - half_h), min(canvas_w, cx + half_w), min(canvas_h, cy + half_h)
        
        # Find min/max coordinates of all tiles
        min_x = min(t.cx - tile_size / 2 for t in tiles)
        max_x = max(t.cx + tile_size / 2 for t in tiles)
        min_y = min(t.cy - tile_size / 2 for t in tiles)
        max_y = max(t.cy + tile_size / 2 for t in tiles)
        
        # Calculate scene size and add padding
        scene_w = max_x - min_x
        scene_h = max_y - min_y
        padding_w = scene_w * padding_frac
        padding_h = scene_h * padding_frac
        
        # Apply padding
        x1 = max(0, int(min_x - padding_w))
        y1 = max(0, int(min_y - padding_h))
        x2 = min(canvas_w, int(max_x + padding_w))
        y2 = min(canvas_h, int(max_y + padding_h))
        
        return x1, y1, x2, y2

    def render_tiles(self, tiles: list[Tile], canvas_w: int, canvas_h: int, tile_size: int, rng: np.random.Generator, out_size=(128, 128), aug_img=False) -> torch.Tensor:
        # Calculate initial bounding box
        x1, y1, x2, y2 = self.calculate_bounding_box(tiles, tile_size, canvas_w, canvas_h, self.crop_padding_frac)
        
        # Apply random crop jitter for augmentation (simulates camera movement)
        if aug_img and self.crop_jitter_frac > 0:
            crop_w = x2 - x1
            crop_h = y2 - y1
            jitter_x = rng.uniform(-crop_w * self.crop_jitter_frac, crop_w * self.crop_jitter_frac)
            jitter_y = rng.uniform(-crop_h * self.crop_jitter_frac, crop_h * self.crop_jitter_frac)
            x1 = int(max(0, x1 + jitter_x))
            y1 = int(max(0, y1 + jitter_y))
            x2 = int(min(canvas_w, x2 + jitter_x))
            y2 = int(min(canvas_h, y2 + jitter_y))
        
        # Render full canvas
        img = np.ones((canvas_h, canvas_w, 1), dtype=np.float32)
        for t in tiles:
            if aug_img:
                dx = rng.uniform(-self.jitter_px, self.jitter_px)
                dy = rng.uniform(-self.jitter_px, self.jitter_px)
            else:
                dx = 0.0
                dy = 0.0
            aug_tile = Tile(t.tile_id, t.cx + dx, t.cy + dy, t.color)
            tile_mask = self.render_tile_mask(aug_tile, tile_size, rng, aug_img=aug_img)
            x_start = int(round(aug_tile.cx - tile_size / 2))
            y_start = int(round(aug_tile.cy - tile_size / 2))
            x_end = x_start + tile_size
            y_end = y_start + tile_size
            if x_start < 0 or y_start < 0 or x_end > canvas_w or y_end > canvas_h:
                continue
            img[y_start:y_end, x_start:x_end, :] *= tile_mask
        
        # Crop to region of interest (camera-like cropping)
        cropped_img = img[y1:y2, x1:x2, :]
        
        # Place cropped image in center of fixed-size canvas (no scaling to preserve tile size)
        crop_h, crop_w = cropped_img.shape[:2]
        output_h, output_w = self.output_size[1], self.output_size[0]
        
        # Create canvas with padding (white background)
        canvas = np.ones((output_h, output_w, 1), dtype=np.float32)
        
        if crop_h > 0 and crop_w > 0:
            # If crop is larger than output, center-crop it
            if crop_h > output_h or crop_w > output_w:
                start_y = max(0, (crop_h - output_h) // 2)
                start_x = max(0, (crop_w - output_w) // 2)
                cropped_img = cropped_img[start_y:start_y + output_h, start_x:start_x + output_w, :]
                crop_h, crop_w = cropped_img.shape[:2]
            
            # Center the cropped image on the canvas
            start_y = (output_h - crop_h) // 2
            start_x = (output_w - crop_w) // 2
            canvas[start_y:start_y + crop_h, start_x:start_x + crop_w, :] = cropped_img
        
        return torch.from_numpy(canvas)

    def render_tile_mask(self, tile: Tile, tile_size: int, rng: np.random.Generator, aug_img=False) -> np.ndarray:
        if aug_img and self.dropout_p > 0.0 and rng.uniform(0.0, 1.0) < self.dropout_p:
            return np.ones((tile_size, tile_size, 1), dtype=np.float32)

        base = np.zeros((tile_size, tile_size, 1), dtype=np.float32)
        # Create coordinate grids
        y_coords, x_coords = np.ogrid[:tile_size, :tile_size]

        if aug_img and self.chip_p > 0.0 and rng.uniform(0.0, 1.0) < self.chip_p:
            frac = rng.uniform(0.0, self.chip_max_frac)
            chip_w = int(tile_size * frac)
            chip_h = int(tile_size * frac)
            # Randomly select one of the four corners
            corner = rng.integers(0, 4)
            if corner == 0:  # Top-left triangle
                mask = (x_coords < chip_w) & (y_coords < chip_h) & (x_coords + y_coords <= chip_w)
            elif corner == 1:  # Top-right triangle
                mask = (x_coords >= tile_size - chip_w) & (y_coords < chip_h) & ((tile_size - x_coords) + y_coords <= chip_w)
            elif corner == 2:  # Bottom-left triangle
                mask = (x_coords < chip_w) & (y_coords >= tile_size - chip_h) & (x_coords + (tile_size - y_coords) <= chip_w)
            else:  # Bottom-right triangle
                mask = (x_coords >= tile_size - chip_w) & (y_coords >= tile_size - chip_h) & ((tile_size - x_coords) + (tile_size - y_coords) <= chip_w)
            
            base[mask, :] = 1

        if aug_img and self.edge_dropout_p > 0.0:
            edge_mask = np.zeros((tile_size, tile_size), dtype=bool)
            edge_width = int(tile_size * 0.1)
            edge_mask[:edge_width, :] = True  # Top edge
            edge_mask[-edge_width:, :] = True  # Bottom edge
            edge_mask[:, :edge_width] = True  # Left edge
            edge_mask[:, -edge_width:] = True  # Right edge

            dropout_mask = (rng.uniform(0.0, 1.0, size=(tile_size, tile_size)) < self.edge_dropout_p) & edge_mask
            base[dropout_mask, :] = 1

        if aug_img and (self.rot_deg_max > 0.0 or self.scale_p > 0.0):
            cx = tile_size / 2
            cy = tile_size / 2

            # Apply inverse transformation: for each output pixel, find source in input
            # Center the coordinates
            x_centered = x_coords - cx
            y_centered = y_coords - cy

            # Apply inverse rotation (rotate by -angle)
            if self.rot_deg_max > 0.0:
                angle = np.deg2rad(rng.uniform(-self.rot_deg_max, self.rot_deg_max))
            else:
                angle = 0.0
            
            cos_a = np.cos(-angle)
            sin_a = np.sin(-angle)
            x_rotated = cos_a * x_centered - sin_a * y_centered
            y_rotated = sin_a * x_centered + cos_a * y_centered

            # Apply inverse scaling (divide by scale)
            if self.scale_p > 0.0:
                scale = 1.0 + rng.uniform(-self.scale_p, self.scale_p)
            else:
                scale = 1.0
            
            x_src = x_rotated / scale + cx
            y_src = y_rotated / scale + cy

            # Sample from original base using source coordinates
            xi = np.round(x_src).astype(int)
            yi = np.round(y_src).astype(int)
            
            # Create output array (default to 1s, meaning no tile)
            result = np.ones((tile_size, tile_size, 1), dtype=np.float32)
            mask = (xi >= 0) & (xi < tile_size) & (yi >= 0) & (yi < tile_size)
            result[mask] = base[yi[mask], xi[mask]]
            return result
        return base

    def load_session(self, path: str|Path) -> Session:
        path = Path(path)
        lines = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
        if len(lines) == 0 or lines[0].get("type") != "metadata":
            raise ValueError(f"Invalid session file: {path}")

        metadata = lines[0]
        goal = metadata.get("goal")
        session_id = metadata.get("session_id")
        start_time = float(metadata.get("start_time"))
        canvas_width = int(metadata.get("canvas_width"))
        canvas_height = int(metadata.get("canvas_height"))
        tile_size = int(metadata.get("tile_size"))
        available_colors = list(metadata.get("available_colors", []))

        session = Session(num_steps=0, goal=goal, session_id=session_id, \
                        start_time=start_time, canvas_width=canvas_width, canvas_height=canvas_height, \
                        tile_size=tile_size, available_colors=available_colors, events=[])

        active_tiles = []
        for line in lines[1:]:
            etype = line.get("type")
            session_id = line.get("session_id")
            timestamp = float(line.get("timestamp"))
            tile_id = int(line.get("tile_id"))
            if etype == Event.MOVE:
                cx = float(line.get("final_x"))
                cy = float(line.get("final_y"))
                init_x = float(line.get("init_x"))
                init_y = float(line.get("init_y"))
            else:
                cx = float(line.get("center_x"))
                cy = float(line.get("center_y"))
                init_x = cx
                init_y = cy
            color = line.get("color")
            tile = Tile(tile_id, cx, cy, color)

            
            if session_id != session.session_id:
                raise ValueError(f"Invalid session id at file: {path}\n Expected: {session.session_id}, Found: {session_id}")

            if etype == Event.ADD:
                active_tiles.append(tile)
            elif etype == Event.MOVE:
                changed = False
                for at in active_tiles:
                    if at.tile_id == tile.tile_id:
                        active_tiles.remove(at)
                        active_tiles.append(tile)
                        changed = True
                if not changed:
                    raise ValueError(f"Invalid session file: {path} for moving tile {tile}")
            elif etype == Event.REMOVE:
                if active_tiles.count(tile) == 0:
                    raise ValueError(f"Invalid session file: {path} for removing tile {tile}")
                else:
                    active_tiles.remove(tile)
            else:
                raise ValueError(f"Invalid event type at file: {path}")

            event = Event(etype, session_id, timestamp, tile, init_x, init_y, active_tiles)
            session.events.append(event)
            session.num_steps += 1
        return session


# For testing on local
if __name__ == "__main__":
    dataset = MosaicDataset("../mosaic_GUI/session_logs", include_intermediate=False, copy_num=5, enable_augmentation=True)
    print(f"Total dataset length: {len(dataset)}")
    dataset.set_epoch(3)
    for i in range(len(dataset)):
        img = dataset[i]
        print(f"Image {i} shape: {img.shape}")
        import matplotlib.pyplot as plt
        plt.imshow(img.detach().cpu(), cmap="gray")
        plt.axis("off")
        plt.show()