"""Default-off options for the bounded full-pool B2 paired-retention candidate (P16 stage 2: native paired checkpoints)."""
from dataclasses import dataclass
import os

# P16 stage 2: the native recurrent checkpoint budget (--recurrent-cache-gib) of the shared-prefix profile. It holds both
# sides of every paired checkpoint (target KDA state ~149.5 MB + DFlash2 window ~41.9 MB, ~191.4 MB). 5 GiB keeps the
# stage-1 checkpoint count (4 GiB of ~149.5 MB target-only checkpoints, 28) and replaces stage 1's 4 GiB native +
# 1 GiB B2 paired budget (--session-cache-gib, retired)
B2_NATIVE_CHECKPOINT_BYTES=5*1024**3
# Qualified B2 page pool and its per-request context (the pool less one reserved page)
B2_CACHE_TOKENS=393216
B2_CONTEXT_TOKENS=B2_CACHE_TOKENS-256
# GLM5.3 Q8 target bytes per MLA layer per page with the full k_idx plane (256 rows of 512 B, c1 L1c); the
# engine k_idx ring keeps W of those rows, so a larger pool holds the same bytes as the qualified one (iso-storage)
FULL_PAGE_BYTES=286720
KIDX_ROW_BYTES=512


def kidx_ring_rows():
    """Engine k_idx ring rows (EXL3_MLA_KIDX_RING, read by exllamav3/cache/mla.py); 0 = full indexer plane."""
    value=int(os.environ.get('EXL3_MLA_KIDX_RING','0') or '0')
    if value<0:raise ValueError('EXL3_MLA_KIDX_RING must not be negative')
    return value


def ring_pool_limit(ring):
    """Largest page-aligned pool (tokens) whose target bytes stay within the qualified 393216-token full-plane
    pool when the engine keeps `ring` k_idx rows per page: 655360 at 32 rows."""
    page=FULL_PAGE_BYTES-(256-ring)*KIDX_ROW_BYTES
    return B2_CACHE_TOKENS*FULL_PAGE_BYTES//page//256*256


def context_cap_option(args):
    """Optional --context-cap: the per-request context ceiling, decoupled from the page pool (-cs). Unset keeps
    the derived cache_size - 256. Never above the qualified 392960 or the pool less its reserved page. Any pool
    above 393216 tokens, B2 or not, needs the engine k_idx ring, stays within the ring's iso-storage size and
    requires the cap (the per-request staging bounds were qualified at 392960)."""
    cap=getattr(args,'context_cap',None)
    size=getattr(args,'cache_size',0)
    if type(size) is int and size>B2_CACHE_TOKENS:
        ring=kidx_ring_rows()
        if not ring or size%256 or size>ring_pool_limit(ring):
            raise ValueError('A pool above 393216 tokens needs EXL3_MLA_KIDX_RING and a page-aligned size within '
                             'its iso-storage limit (655360 at 32 rows)')
        if cap is None:
            raise ValueError('A pool above 393216 tokens requires --context-cap of at most 392960')
    if cap is None:return None
    if type(cap) is not int or type(size) is not int or not 256<=cap<=min(B2_CONTEXT_TOKENS,size-256):
        raise ValueError('--context-cap must be an integer from 256 through min(392960, cache_size - 256)')
    return cap


def cache_geometry(cache):
    """Static target page geometry for /health: pool pages, engine k_idx ring rows and target cache bytes per
    page summed over layers and ranks (one rank's slot share is in the tier stats)."""
    layers=list(getattr(cache,'layers',{}).values())
    pages=cache.max_num_tokens//256
    if cache.max_num_tokens>B2_CACHE_TOKENS:
        # After load: every k-pool layer the engine built must carry the ring the options were checked against
        rings={getattr(layer,'kidx_ring',0) for layer in layers if getattr(layer,'kpool',0)}
        if rings!={kidx_ring_rows()} or 0 in rings:
            raise ValueError('A pool above 393216 tokens was loaded without the engine k_idx ring on every k-pool layer')
    return dict(cache_pages=pages,
                kidx_ring_rows=max((getattr(layer,'kidx_ring',0) for layer in layers),default=0),
                target_page_bytes=sum(int(layer.storage_size()) for layer in layers)//pages if layers else None)


def _b2_pool_ok(args):
    """The exact 393216 pool, or a larger page-aligned pool when the engine's k_idx ring frees the bytes and an
    explicit --context-cap keeps every request inside the qualified per-request context."""
    size=getattr(args,'cache_size',0)
    if size==B2_CACHE_TOKENS:return True
    return (type(size) is int and size>B2_CACHE_TOKENS and size%256==0 and kidx_ring_rows()>0
            and context_cap_option(args) is not None)


@dataclass(frozen=True)
class B2RetentionOptions:
    enabled: bool = False
    target_cpu_bytes: int = 0
    shared_prefix_tokens: int = 0


def b2_retention_options(args, concurrency, retention):
    enabled=getattr(args,'experimental_b2_session_cache',False)
    target=getattr(args,'b2_target_cpu_cache_gib',None)
    shared=getattr(args,'experimental_b2_shared_prefix',False)
    prefix=getattr(args,'b2_shared_prefix_tokens',None)
    if type(shared) is not bool or (prefix is not None and not shared):
        raise ValueError('B2 shared option must be explicit; cap requires its flag')
    if shared and not enabled:
        raise ValueError('B2 shared requires its paired owner')
    if shared:
        prefix=65536 if prefix is None else prefix
        # The optional target CPU tier keeps its own 0..8 GiB check below. P16 stage 2: the native recurrent cache
        # holds both sides of every checkpoint (B2's own 1 GiB / eight-pair budget is gone).
        if (type(prefix) is not int or not 256<=prefix<=393216 or prefix%256
                or retention.recurrent_bytes!=B2_NATIVE_CHECKPOINT_BYTES):
            raise ValueError('B2 shared starts with the exact native 5GiB paired checkpoint budget and a bounded page cap')
    if type(enabled) is not bool:
        raise ValueError('Experimental B2 session flag must be a bool')
    if not enabled:
        if target is not None:
            raise ValueError('B2 target budget requires --experimental-b2-session-cache')
        return B2RetentionOptions()
    target=0 if target is None else target
    if type(target) is not int or not 0<=target<=8:
        raise ValueError('B2 target CPU budget must be an integer GiB from 0 through 8')
    if (not concurrency.enabled or concurrency.max_sequences not in (2,4)
            or not _b2_pool_ok(args)
            or getattr(args,'num_draft_tokens',None)!=7
            or getattr(args,'tensor_parallel',False) is not True
            or getattr(args,'experimental_dflash2_tp',False) is not True
            or retention.enabled or retention.target_cpu_bytes
            or getattr(args,'dflash_prefix_cache',False)
            or getattr(args,'cpu_cache_size',0)):
        raise ValueError('B2 session candidate requires exact 393216-token TP/K7 (a larger pool only with '
                         'EXL3_MLA_KIDX_RING and --context-cap), two or four sequences and its own paired owner')
    return B2RetentionOptions(True,target*1024**3,prefix if shared else 0)
