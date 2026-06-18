import numpy as np
import gymnasium as gym
import torch
import json
import random
import string
from pathlib import Path
from typing import List, Dict, Any
from collections import OrderedDict
from gymnasium import spaces
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sentence_transformers import SentenceTransformer
import nltk
from nltk.corpus import words as nltk_words

nltk.download('words', quiet=True)
EN_WORDS = [w for w in nltk_words.words() if 4 <= len(w) <= 12]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device = "cpu"

SUMMARY_KEYS = [
    "files", "read_files", "write_files", "delete_files",
    "keys", "read_keys", "write_keys", "delete_keys",
    "executed_commands", "resolved_apis", "mutexes",
    "created_services", "started_services"
]

SEP = "\\"  # path-like separator used in replacement logic

class RLattacker(gym.Env):
    def __init__(self,
                 model_dir: str = "models/distilbert",
                 data_root: str = "data/malware_dataset/vobfus",
                 steps: int = 1000,
                 seed: int = 42,
                 threshold: float = 0.5,
                 offset: int = 0,
                 rew: int = 1,
                 pomdp: bool = False):
        super().__init__()
        random.seed(seed)
        np.random.seed(seed)

        self.steps = steps
        self.threshold = threshold
        self.reward_mode = rew
        self.pomdp = pomdp
        self.device = device
        self.model_dir = Path(model_dir)
        self.data_root = Path(data_root)

        # Load model & tokenizer.
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_dir, trust_remote_code=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token if self.tokenizer.eos_token else self.tokenizer.unk_token

        adapter_cfg_path = self.model_dir / "adapter_config.json"
        if adapter_cfg_path.exists():
            from peft import PeftModel
            with adapter_cfg_path.open() as _f:
                _adapter_cfg = json.load(_f)
            base = AutoModelForSequenceClassification.from_pretrained(
                _adapter_cfg["base_model_name_or_path"], num_labels=2
            )
            self.model = PeftModel.from_pretrained(base, self.model_dir).to(self.device).eval()
        else:
            self.model = AutoModelForSequenceClassification.from_pretrained(
                self.model_dir
            ).to(self.device).eval()

        # Collect report paths
        self.report_paths = sorted(self.data_root.rglob("*.json"))

        # Resolve repository root
        self.repo_root = Path(__file__).resolve().parent.parent

        # Load goodware corpus (weighted terms) directly from corpus.json
        raw_corpus = json.loads((self.repo_root / "corpus.json").read_text(encoding="utf-8"))
        self.corpus_freq = {str(k): int(v) for k, v in raw_corpus.items()}
        self.corpus_tokens = list(self.corpus_freq.keys()) if self.corpus_freq else ["token"]
        self.corpus_weights = [self.corpus_freq[t] for t in self.corpus_tokens]
        
        # Load goodware entries pool template
        sum_path = self.repo_root / "sum.json"
        raw_sum = json.loads(sum_path.read_text(encoding="utf-8"))
        self.goodware_pool_template = {k: list(v) for k, v in raw_sum.items()}

        # Action space: (operation, key index, generation mode)
        self.action_space = spaces.MultiDiscrete([2, 13, 4])
        
       # Different observation space for fully / partially observable mode
        if self.pomdp:
            self.sentence_model = SentenceTransformer('all-MiniLM-L6-v2', device=str(self.device))
            self.embedding_dim_pomdp = self.sentence_model.get_sentence_embedding_dimension()
            self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(self.embedding_dim_pomdp,), dtype=np.float32)
        else:
            self.embedding_dim = 768
            self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(self.embedding_dim,), dtype=np.float32)        


        # Initializations
        self.curr_index = offset
        self.resets = 0
        self.done = False
        self.iters = []
        self.probs = []
        self.rews = []

        # Chunk-level cache: maps chunk_text -> (probs, embedding)
        self._chunk_cache: OrderedDict[str, tuple] = OrderedDict()
        self._cache_max_size = 5000  # Limit cache to prevent memory bloat

    def load_report(self, max_tokens: int = 8000) -> None:
        while self.curr_index < len(self.report_paths):
            path = self.report_paths[self.curr_index-1]
            raw = json.loads(path.read_text(encoding='utf-8'))
            summary = raw.get("summary") if isinstance(raw, dict) else None
            if not isinstance(summary, dict):
                raise ValueError(f"Report {path} missing summary section")
            
            # Normalize order/keys per SUMMARY_KEYS
            temp_report = {key: list(summary.get(key, [])) for key in SUMMARY_KEYS}
            
            # Check token length
            text = json.dumps(temp_report, separators=(",", ":"))
            tokens = self.tokenizer.encode(text, add_special_tokens=False, truncation=False)
            
            if len(tokens) <= max_tokens:
                # Report is acceptable, load it
                self.report = temp_report
                self.original_lengths = [len(self.report[k]) for k in SUMMARY_KEYS]
                return
            else:
                self.curr_index += 1
        # Raise error if report are exhausted
        raise ValueError(f"No valid reports found (all exceed {max_tokens} tokens)")

    def classify(self) -> np.ndarray:
        """
        Classify the current report using chunking for reports larger than max_length.
        We split these reports into chunks, classify each chunk, and aggregate both
        probabilities (via averaging) and embeddings (via mean pooling). Uses
        chunk-level caching: only chunks that changed are re-classified.
        
        Returns:
            np.ndarray: [goodware_prob, malware_prob]
        """
        # Tokenize the full text to get all tokens
        text = json.dumps(self.report, separators=(",", ":"))
        # Dont truncate, long sequences are handled via chunking
        full_tokens = self.tokenizer.encode(text, add_special_tokens=False, truncation=False)
        
        # Chunking parameters
        max_length = 512
        overlap = 20
        
        # Build chunks
        chunks = []
        if len(full_tokens) <= max_length:
            chunks = [text]
        else:
            # Split into overlapping chunks
            start = 0
            while start < len(full_tokens):
                end = min(start + max_length, len(full_tokens))
                chunk_tokens = full_tokens[start:end]
                chunk_text = self.tokenizer.decode(chunk_tokens, skip_special_tokens=True)
                chunks.append(chunk_text)

                if end >= len(full_tokens):
                    break
                start += max_length - overlap
        
        # Separate cached and uncached chunks
        uncached_chunks = []
        uncached_indices = []
        all_probs = [None] * len(chunks)
        all_embeddings = [None] * len(chunks) if not self.pomdp else None
        
        for i, chunk in enumerate(chunks):
            if chunk in self._chunk_cache:
                # Use cached result
                cached_probs, cached_emb = self._chunk_cache[chunk]
                all_probs[i] = cached_probs
                if not self.pomdp:
                    all_embeddings[i] = cached_emb
                # Move to end (mark as recently used)
                self._chunk_cache.move_to_end(chunk)
            else:
                # Mark for batch processing
                uncached_chunks.append(chunk)
                uncached_indices.append(i)
        
        # Batch process only uncached chunks
        if uncached_chunks:
            inputs = self.tokenizer(
                uncached_chunks,
                truncation=True,
                max_length=max_length,
                padding='max_length',
                return_tensors='pt'
            ).to(self.device)
            
            with torch.no_grad():
                # Get model outputs, with hidden states if in MDP mode
                if self.pomdp:
                    outputs = self.model(**inputs)
                else:
                    outputs = self.model(**inputs, output_hidden_states=True)
                
                # Extract probabilities for all chunks
                chunk_probs = torch.softmax(outputs.logits, dim=1).cpu().numpy()
                
                # Extract [CLS] embeddings (in MDP mode)
                if not self.pomdp:
                    last_hidden_state = outputs.hidden_states[-1]  # (batch, seq_len, hidden_dim)
                    cls_embeddings = last_hidden_state[:, 0, :].cpu().numpy()  # (batch, hidden_dim)
            
            # Store results and update cache
            for idx, chunk_idx in enumerate(uncached_indices):
                probs = chunk_probs[idx]
                all_probs[chunk_idx] = probs
                
                chunk_key = uncached_chunks[idx]
                
                # Evict oldest entry if cache is full (O(1) operation)
                if len(self._chunk_cache) >= self._cache_max_size:
                    self._chunk_cache.popitem(last=False)
                
                if self.pomdp:
                    # Cache without embedding
                    self._chunk_cache[chunk_key] = (probs, None)
                else:
                    # Cache with embedding
                    emb = cls_embeddings[idx]
                    all_embeddings[chunk_idx] = emb
                    self._chunk_cache[chunk_key] = (probs, emb)
        
        # Aggregate results across all chunks
        # Average probabilities across all chunks
        probs = np.mean(all_probs, axis=0)
        
        if self.pomdp:
            # POMDP mode: return probabilities and sentence embedding
            sent_emb = self.get_sentence_embedding()
            return (probs, sent_emb)
        else:
            # MDP mode: return probabilities and embeddings
            # Mean pooling of embeddings across all chunks
            embedding = np.mean(all_embeddings, axis=0)
            # Store the current embedding
            self.current_embedding = embedding
            return (probs, embedding)

    def get_sentence_embedding(self) -> np.ndarray:
        # Compute sentence transformer embedding of the current report.
        text = json.dumps(self.report, separators=(",", ":"))
        embedding = self.sentence_model.encode(text, convert_to_numpy=True, show_progress_bar=False)
        return embedding.astype(np.float32)

    def get_word(self, mode: int) -> str:
        if mode == 3:
            mode = random.choice([0, 1, 2])
        if mode == 0:
            return random.choice(EN_WORDS)
        if mode == 1:
            return random.choices(self.corpus_tokens, weights=self.corpus_weights, k=1)[0]
        if mode == 2:
            lgth = random.randrange(4, 10)
            return ''.join(random.choices(string.ascii_letters + string.digits, k=lgth))

    def replace_entry(self, entry: str, mode: int) -> str:
        parts = entry.split(SEP)
        if len(parts) > 2:
            for i in range(2, len(parts)):
                parts[i] = self.get_word(mode)
            return SEP.join(parts)
        elif len(parts) == 1:
            return self.get_word(mode)
        else:
            return self.get_word(mode) + SEP + self.get_word(mode)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.done = False
        self.iter, self.adds, self.mods = 0, 0, 0
        self.accrew = 0.0
        self.resets += 1

        self.original_lengths: List[int] = []
        # Current report summary (flat dict: key -> list[str])
        self.report: Dict[str, List[str]] = {key: [] for key in SUMMARY_KEYS}
        self.edit_positions = [0] * len(SUMMARY_KEYS)

        # Create fresh copy of goodware pool from template
        self.goodware_pool = {k: list(v) for k, v in self.goodware_pool_template.items()}
        
        self.curr_index += 1 # Increment current index
        self.load_report()
        
        self.edit_positions = [0] * len(SUMMARY_KEYS)
        self._chunk_cache.clear()  # Clear chunk cache on reset
        probs, emb = self.classify()
        # print(f"Class probs {probs}")
        self.prev_mal_prob = probs[1]
        self.mal_prob = probs[1]

        # Return observation based on MDP/POMDP mode
        if self.pomdp:
            obs = emb.astype(np.float32)  # Sentence transformer embedding
        else:
            obs = emb.astype(np.float32)  # Classifier embedding
        info = {
            "iter": 0,
            "adds": 0,
            "mods": 0,
            "mal_prob": float(self.mal_prob)
        }

        return obs, info

    def step(self, action):
        if self.done:
            # Return current observation based on mode
            if self.pomdp:
                obs = self.get_sentence_embedding()
            else:
                obs = self.current_embedding.astype(np.float32)
            return obs, 0.0, True, False, {}
        op, key_idx, gen_mode = action
        key = SUMMARY_KEYS[key_idx]

        if op == 0:  # Adding a goodware entry from pool for the selected key
            pool = self.goodware_pool.get(key, [])
            if pool:
                new_entry = pool.pop(0)  # consume from the top (first entry)
                self.report[key].append(new_entry)
                self.adds += 1
                # No need to invalidate cache - chunk-level cache handles changes automatically
        elif op == 1:  # Edit single entry
            if key != "resolved_apis": # Resolved_apis cannot be modified
                pos = self.edit_positions[key_idx]
                if pos < len(self.report[key]):
                    self.report[key][pos] = self.replace_entry(self.report[key][pos], gen_mode)
                    self.mods += 1
                    # No need to invalidate cache - chunk-level cache handles changes automatically
                self.edit_positions[key_idx] += 1
        else:
            raise ValueError("Invalid operation ID")

        self.iter += 1
        probs, emb = self.classify()
        self.prev_mal_prob, self.mal_prob = self.mal_prob, probs[1]

        # Termination conditions
        terminated = self.mal_prob < self.threshold  # Task succeeded
        truncated = self.iter >= self.steps  # Step limit reached
        self.done = terminated or truncated
        
        bonus = 100/self.iter if terminated else 0.0
        reward = self.compute_reward(bonus)
        self.accrew += reward

        if self.done:
            # print(self.mal_prob, self.iter)
            self.probs.append(self.mal_prob)
            self.iters.append(self.iter)
            self.rews.append(self.accrew)
        # Return observation based on MDP/POMDP mode
        if self.pomdp:
            obs = emb.astype(np.float32)  # Sentence transformer embedding
        else:
            obs = emb.astype(np.float32)  # Classifier embedding
        info = {
            "iter": self.iter,
            "adds": self.adds,
            "mods": self.mods,
            "mal_prob": float(self.mal_prob),
            "prev_mal_prob": float(self.prev_mal_prob),
            "done": self.done,
        }
        return obs, reward, terminated, truncated, info

    # Reward Logic 
    def reward1(self):
        # Positive reward when malware probability decreases
        return self.prev_mal_prob - self.mal_prob

    def reward2(self, bonus):
        # Get bonus based on how fast the agent succeeds
        return self.reward1() + bonus

    def compute_reward(self, bonus):
        if self.reward_mode == 1:
            return self.reward1()
        if self.reward_mode == 2:
            return self.reward2(bonus)

    def results(self):    
        return [np.mean(self.probs), self.iters, self.rews, self.resets]