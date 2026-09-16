"""Opt-in shared replay hash for MCPGate args_hash_v=1; legacy canonical.py remains unchanged."""
import hashlib
import json
import math
import re
from decimal import ROUND_HALF_UP, Decimal, localcontext

DROP = {'request_id', 'trace_id', 'timestamp', 'ts', 'cursor', 'page_token', 'nonce'}

def normalize(v):
    if isinstance(v, dict):
        return {k: normalize(x) for k, x in v.items() if k not in DROP}
    if isinstance(v, list):
        return [normalize(x) for x in v]
    if isinstance(v, float):
        if not math.isfinite(v):
            raise ValueError('Nonfinite number')
        with localcontext() as ctx:
            ctx.prec = 100
            if v.is_integer():
                return v
            return float(Decimal.from_float(v).quantize(Decimal('0.000001'), rounding=ROUND_HALF_UP))
    if isinstance(v, str):
        v = v.strip()
        if re.fullmatch(r'\d{4}-\d{2}-\d{2}(?:[T ]\d{2}:\d{2}(?::\d{2})?(?:\.\d+)?(?:Z|z|[+-]\d{2}:?\d{2})?)?', v):
            return '<ts>'
        if re.fullmatch(r'[0-9a-fA-F]{8}(?:-[0-9a-fA-F]{4}){3}-[0-9a-fA-F]{12}', v):
            return '<uuid>'
    return v

def canonical(v):
    if isinstance(v, dict):
        keys = sorted(v, key=lambda k: k.encode())
        body = ','.join(json.dumps(k, ensure_ascii=False) + ':' + canonical(v[k]) for k in keys)
        return '{' + body + '}'
    if isinstance(v, list):
        return '[' + ','.join(canonical(x) for x in v) + ']'
    if isinstance(v, (int, float)) and not isinstance(v, bool):
        if v == 0:
            return '0'
        if abs(v) >= 1e21:
            return json.dumps(v, separators=(',', ':'))
        if int(v) == v:
            return str(int(v))
        return format(v, '.6f').rstrip('0').rstrip('.')
    return json.dumps(v, ensure_ascii=False, separators=(',', ':'))


ARGS_HASH_VERSION = 1

def args_hash(tool, args):
    return hashlib.sha256(canonical({'tool': tool, 'args': normalize(args)}).encode('utf-8')).hexdigest()
