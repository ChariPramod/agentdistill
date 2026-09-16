"""Shared MCPGate migration vectors, independent of legacy v1 replay data."""
import json
import unittest
from pathlib import Path

from agentdistill.canonical_shared import args_hash, canonical, normalize


class SharedCanonicalTests(unittest.TestCase):
    def test_shared_vectors(self):
        data = json.loads((Path(__file__).resolve().parents[1] / 'schemas/canonical-shared-vectors.json').read_text())
        for v in data['vectors']:
            with self.subTest(v['name']):
                self.assertEqual(normalize(v['args']), v['normalized'])
                self.assertEqual(canonical({'tool':v['tool'], 'args':normalize(v['args'])}), v['canonical'])
                self.assertEqual(args_hash(v['tool'],v['args']), v['args_hash'])
