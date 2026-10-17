import numpy as np
import torch


class ShardedTokens:
    """Many uint16 .bin files viewed as one pool of non-overlapping training blocks.

    A block is block_size + 1 tokens (inputs = first block_size, targets = shifted by one);
    consecutive blocks share only that one boundary token.
    """

    def __init__(self, files, block_size):
        if not files:
            raise ValueError("No data files given.")
        self.block_size = block_size
        self.maps = [np.memmap(f, dtype=np.uint16, mode="r") for f in files]
        counts = [(len(m) - 1) // block_size for m in self.maps]
        self.cum = np.concatenate([[0], np.cumsum(counts)]).astype(np.int64)
        self.n_blocks = int(self.cum[-1])
        if self.n_blocks == 0:
            raise ValueError("Data files are too small for this block size.")

    def __len__(self):
        return self.n_blocks

    def get(self, i):
        s = int(np.searchsorted(self.cum, i, side="right") - 1)
        off = int(i - self.cum[s]) * self.block_size
        chunk = np.asarray(self.maps[s][off: off + self.block_size + 1], dtype=np.int64)
        return chunk[:-1], chunk[1:]


class BatchLoader:
    """Deterministic shuffled batches. The only state is `pos` (samples consumed), so a
    resumed run continues exactly where it stopped. Each epoch uses a new permutation
    seeded by (seed + epoch).

    Do NOT add or remove shards between a checkpoint and its resume: the permutation
    depends on the total number of blocks.
    """

    def __init__(self, tokens, batch_size, seed=1337):
        self.tokens = tokens
        self.batch_size = batch_size
        self.seed = seed
        self.pos = 0
        self._epoch = -1
        self._order = None

    @property
    def n_blocks(self):
        return self.tokens.n_blocks

    @property
    def epoch(self):
        return self.pos // self.tokens.n_blocks

    def _set_epoch(self, epoch):
        rng = np.random.default_rng(self.seed + epoch)
        self._order = rng.permutation(self.tokens.n_blocks)
        self._epoch = epoch

    def next_batch(self):
        n = self.tokens.n_blocks
        xs, ys = [], []
        for _ in range(self.batch_size):
            epoch, p = divmod(self.pos, n)
            if epoch != self._epoch:
                self._set_epoch(epoch)
            x, y = self.tokens.get(int(self._order[p]))
            xs.append(x)
            ys.append(y)
            self.pos += 1
        return torch.from_numpy(np.stack(xs)), torch.from_numpy(np.stack(ys))

    def state_dict(self):
        return {"pos": self.pos, "n_blocks": self.tokens.n_blocks, "seed": self.seed}

    def load_state_dict(self, state):
        if state.get("n_blocks") != self.tokens.n_blocks:
            print(f"WARNING: shard set changed (checkpoint had {state.get('n_blocks')} blocks, "
                  f"now {self.tokens.n_blocks}). Data order will not match the original run.")
        self.pos = int(state["pos"])
        self._epoch = -1
