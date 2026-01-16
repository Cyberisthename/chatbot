"""src/bio_knowledge/protein_sequence_retriever.py

Protein Sequence Retrieval (UniProt)

This module retrieves REAL protein sequences for UniProt accessions.

Primary mode:
- Fetch from UniProt REST API (https://rest.uniprot.org/) and cache locally.

Offline mode:
- If the sequence is already cached, it will be used without network access.

This is designed to eliminate placeholder/synthetic sequences for quantum
protein analysis while keeping the system usable in offline environments
once data has been cached.
"""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional


@dataclass
class ProteinSequenceRecord:
    uniprot_id: str
    sequence: str
    fetched_at: float
    source_url: str

    def to_dict(self) -> Dict:
        return {
            "uniprot_id": self.uniprot_id,
            "sequence": self.sequence,
            "fetched_at": self.fetched_at,
            "source_url": self.source_url,
        }


class ProteinSequenceRetriever:
    def __init__(self, cache_dir: str = "./protein_sequence_cache"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _cache_path(self, uniprot_id: str) -> Path:
        return self.cache_dir / f"{uniprot_id}.json"

    def get_sequence(self, uniprot_id: str, allow_network: bool = True) -> Optional[ProteinSequenceRecord]:
        uniprot_id = uniprot_id.strip()
        cached = self._read_cache(uniprot_id)
        if cached is not None:
            return cached

        if not allow_network:
            return None

        return self._fetch_and_cache(uniprot_id)

    def _read_cache(self, uniprot_id: str) -> Optional[ProteinSequenceRecord]:
        path = self._cache_path(uniprot_id)
        if not path.exists():
            return None

        data = json.loads(path.read_text())
        seq = data.get("sequence", "")
        if not seq:
            return None

        return ProteinSequenceRecord(
            uniprot_id=data.get("uniprot_id", uniprot_id),
            sequence=seq,
            fetched_at=float(data.get("fetched_at", 0.0)),
            source_url=data.get("source_url", ""),
        )

    def _fetch_and_cache(self, uniprot_id: str) -> Optional[ProteinSequenceRecord]:
        url = f"https://rest.uniprot.org/uniprotkb/{uniprot_id}.fasta"
        req = urllib.request.Request(url, headers={"User-Agent": "cto.new-cancer-research/1.0"})
        try:
            with urllib.request.urlopen(req, timeout=20) as resp:
                fasta = resp.read().decode("utf-8")
        except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError):
            return None

        sequence = self._parse_fasta_sequence(fasta)
        if not sequence:
            return None

        record = ProteinSequenceRecord(
            uniprot_id=uniprot_id,
            sequence=sequence,
            fetched_at=time.time(),
            source_url=url,
        )

        self._cache_path(uniprot_id).write_text(json.dumps(record.to_dict(), indent=2))
        return record

    @staticmethod
    def _parse_fasta_sequence(fasta: str) -> str:
        lines = [ln.strip() for ln in fasta.splitlines() if ln.strip()]
        seq_lines = [ln for ln in lines if not ln.startswith(">")]
        return "".join(seq_lines).strip()
