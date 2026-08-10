"""
Apple NLU — Native NaturalLanguage.framework via Swift subprocess.
No ObjC bridge. No pyobjc. Pure Swift binary, JSON-line protocol.
"""

import subprocess
import json
import os
import numpy as np
from typing import Optional, Tuple, List, Dict

_DIR = os.path.dirname(os.path.abspath(__file__))
_ENGINE = os.path.join(_DIR, '..', 'nlu', 'nlu_engine')
_NLU_DIR = os.path.join(_DIR, '..', 'nlu')


class AppleNLU:
    """Native Apple NaturalLanguage.framework.

    Swift subprocess stays alive — models stay loaded in ANE SRAM.
    """

    def __init__(self):
        if not os.path.exists(_ENGINE):
            raise FileNotFoundError(
                f"NLU engine not built. Run: cd nlu && bash build.sh"
            )

        self._proc = subprocess.Popen(
            [_ENGINE, _NLU_DIR],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
        )

        ready_line = self._proc.stdout.readline()
        if not ready_line:
            err = self._proc.stderr.read()
            raise RuntimeError(f"NLU engine failed: {err}")

        self._info = json.loads(ready_line)
        if not self._info.get('ready'):
            raise RuntimeError(f"NLU not ready: {self._info}")

        self.dim = self._info['dim']
        self.contextual = self._info.get('contextual', False)
        self.has_classifier = self._info.get('classifier', False)
        self.languages = self._info.get('languages', [])

    def _call(self, cmd: dict) -> dict:
        line = json.dumps(cmd, ensure_ascii=False)
        self._proc.stdin.write(line + '\n')
        self._proc.stdin.flush()
        resp = self._proc.stdout.readline()
        if not resp:
            raise RuntimeError("NLU engine died")
        return json.loads(resp)

    def embed(self, text: str) -> Optional[np.ndarray]:
        """Sentence embedding (L2-normalized). Returns None on failure."""
        r = self._call({"cmd": "embed", "text": text})
        if r.get('ok'):
            return np.array(r['vec'], dtype=np.float32)
        return None

    def similarity(self, a: str, b: str) -> float:
        """Cosine similarity between two texts."""
        r = self._call({"cmd": "sim", "a": a, "b": b})
        return r.get('score', 0.0)

    def language(self, text: str) -> Tuple[str, float]:
        """Detect language. Returns (code, confidence)."""
        r = self._call({"cmd": "lang", "text": text})
        return r.get('lang', 'en'), r.get('confidence', 0.0)

    def classify(self, text: str) -> Tuple[Optional[str], float, Dict[str, float]]:
        """Intent classification. Returns (label, confidence, all_scores)."""
        r = self._call({"cmd": "classify", "text": text})
        if r.get('ok'):
            return r['label'], r['confidence'], r.get('scores', {})
        return None, 0.0, {}

    def lemmatize(self, text: str) -> List[Dict[str, str]]:
        """Lemmatize text. Returns [{text, lemma, pos}, ...]."""
        r = self._call({"cmd": "lemma", "text": text})
        return r.get('tokens', [])

    def close(self):
        try:
            self._call({"cmd": "quit"})
        except Exception:
            pass
        self._proc.terminate()

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass

    def __repr__(self):
        return (f"AppleNLU(dim={self.dim}, contextual={self.contextual}, "
                f"classifier={self.has_classifier}, langs={self.languages})")


_instance = None

def get_nlu() -> AppleNLU:
    global _instance
    if _instance is None:
        _instance = AppleNLU()
    return _instance
