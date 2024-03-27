class Tokenizer:
    def __init__(self, num_merges, data):
        self.num_merges = num_merges
        self.vocab = None
        self.merges = None
        self.vocab, self.merges = self.train_tokenizer(data, num_merges)

    def get_stats(self, ids, counts=None):
        if counts is None:
            counts = {}
        for pair in zip(ids, ids[1:]):
            if pair not in counts:
                counts[pair] = 0
            counts[pair] += 1
        return counts 

    # %%
    def merge(self,ids, pair, idx):
        new_ids = []
        i = 0
        while i < len(ids):
            if ids[i] == pair[0] and i < len(ids) - 1 and ids[i + 1] == pair[1]:
                new_ids.append(idx)
                i += 2
            else:
                new_ids.append(ids[i])
                i += 1
        return new_ids


    # %%
    # now put it all together into a tokenizer function
    def train_tokenizer(self,data, num_merges):
        text_bytes = data.encode('utf-8')
        ids = list(text_bytes)

        merges = {} # (int, int) -> int
        vocab = {idx: bytes([idx]) for idx in range(256)} # int -> bytes 

        for i in range(num_merges):
            stats = self.get_stats(ids)
            pair = max(stats, key=stats.get)
            idx = 256 + i
            ids = self.merge(ids, pair, idx)
            merges[pair] = idx
            vocab[idx] = vocab[pair[0]] + vocab[pair[1]]

        return vocab, merges

    # %%
    def encode(self, text):
        ids = list(text.encode('utf-8'))
        i = 0
        while i < len(ids):
            if i < len(ids) - 1 and (ids[i], ids[i + 1]) in self.merges:
                ids = self.merge(ids, (ids[i], ids[i + 1]), self.merges[(ids[i], ids[i + 1])])
            i += 1
        return ids

    def decode(self, ids):
        text = b""
        for i in ids:
            text += self.vocab[i]
        return text.decode('utf-8', errors='replace')


