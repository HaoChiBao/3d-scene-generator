"""Patch WorldMirror attention to fall back to SDPA when flash_attn is absent."""

from pathlib import Path
import sys

REPO = Path(sys.argv[1] if len(sys.argv) > 1 else "/opt/HY-World-2.0")
path = REPO / "hyworld2/worldrecon/hyworldmirror/models/layers/attention.py"
text = path.read_text(encoding="utf-8")

old = """try:
    from flash_attn_interface import flash_attn_func as flash_attn_func_v3
    _USE_FLASH_ATTN_V3 = True
except ImportError:
    from flash_attn.flash_attn_interface import flash_attn_func as flash_attn_func_v2
    _USE_FLASH_ATTN_V3 = False"""

new = """try:
    from flash_attn_interface import flash_attn_func as flash_attn_func_v3
    _USE_FLASH_ATTN_V3 = True
    _HAS_FLASH_ATTN = True
except ImportError:
    try:
        from flash_attn.flash_attn_interface import flash_attn_func as flash_attn_func_v2
        _USE_FLASH_ATTN_V3 = False
        _HAS_FLASH_ATTN = True
    except ImportError:
        flash_attn_func_v3 = None
        flash_attn_func_v2 = None
        _USE_FLASH_ATTN_V3 = False
        _HAS_FLASH_ATTN = False"""

if old not in text:
    raise SystemExit(f"import patch anchor not found in {path}")
text = text.replace(old, new, 1)

old2 = """            if _USE_FLASH_ATTN_V3:
                x = flash_attn_func_v3(q, k, v)
            else:
                x = flash_attn_func_v2(q, k, v, dropout_p=self.attn_drop.p if self.training else 0.0)"""

new2 = """            if _HAS_FLASH_ATTN and _USE_FLASH_ATTN_V3:
                x = flash_attn_func_v3(q, k, v)
            elif _HAS_FLASH_ATTN:
                x = flash_attn_func_v2(q, k, v, dropout_p=self.attn_drop.p if self.training else 0.0)
            else:
                q2, k2, v2 = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
                x = F.scaled_dot_product_attention(q2, k2, v2, dropout_p=self.attn_drop.p if self.training else 0.0)
                x = x.transpose(1, 2)"""

if old2 not in text:
    raise SystemExit(f"apply patch anchor not found in {path}")
text = text.replace(old2, new2, 1)
path.write_text(text, encoding="utf-8")
print(f"Patched flash_attn fallback in {path}")
