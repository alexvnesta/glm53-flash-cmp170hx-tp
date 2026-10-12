"""Pure exact chat-boundary derivation and B2 per-job admission.

This module never accepts an API cache key or a caller-supplied boundary. It uses
the selected live template/tokenizer and binds its exact full token sequence to
the subsequently enqueued job. Native pages, recurrent stashes and raw draft
snapshots remain under the completed manager's existing ownership and budgets.
"""
from contextlib import contextmanager
import copy
import hashlib
import json

PAGE_SIZE = 256
MAX_RENDER_BYTES = 8 << 20
MAX_TOKENS = 400000


def _encoded(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()


def _sha(raw):
    return hashlib.sha256(raw).hexdigest()


def _ids(value):
    if (not isinstance(value, list) or len(value) > MAX_TOKENS
            or not all(type(t) is int and t >= 0 for t in value)):
        raise ValueError("Exact bounded integer token IDs required")
    return value


def _lcp(first, second):
    return next((i for i, (a, b) in enumerate(zip(first, second)) if a != b),
                min(len(first), len(second)))


def _rendered(value):
    if not isinstance(value, str) or len(value.encode()) > MAX_RENDER_BYTES:
        raise ValueError("Bounded exact rendered string required")
    return value


def derive_boundary(messages, tools, effort, prompt, actual_ids, render, tokenize, token_cap):
    """Full-chat renders preserve final suffix; exact token LCP removes BPE/EOS guesses."""
    if not isinstance(messages, list) or not 0 < len(messages) <= 100000:
        return {"admitted": False, "reason": "unsupported_messages"}
    if (not isinstance(tools, list) or len(tools) > 1024
            or any(not isinstance(m, dict) for m in messages)):
        return {"admitted": False, "reason": "unsupported_tools_or_messages"}
    index = next((i for i, m in enumerate(messages) if m.get("role") == "user"), None)
    if index is None or not isinstance(messages[index].get("content"), str):
        return {"admitted": False, "reason": "first_user_requires_text"}
    actual = _rendered(prompt)
    actual_ids = _ids(actual_ids)
    prepared = _encoded([messages, tools, effort])
    if not actual_ids or len(prepared) > MAX_RENDER_BYTES:
        return {"admitted": False, "reason": "prepared_chat_over_cap_or_empty"}
    seed = _sha(prepared)
    prefixes, alternate_ids = [], []
    for suffix in ("A", "B"):
        marker = "__GLM_SHARED_PREFIX_BOUNDARY_" + seed + "_" + suffix + "__"
        if marker in actual:
            return {"admitted": False, "reason": "marker_collision"}
        alternative = copy.deepcopy(messages)
        alternative[index]["content"] = marker
        value = _rendered(render(alternative, effort, copy.deepcopy(tools)))
        if value.count(marker) != 1:
            return {"admitted": False, "reason": "marker_not_unique"}
        prefixes.append(value[:value.index(marker)])
        alternate_ids.append(_ids(tokenize(value)))
    prefix = prefixes[0]
    if prefixes[1] != prefix or not actual.startswith(prefix):
        return {"admitted": False, "reason": "renderer_prefix_not_invariant"}
    prefix_ids = _ids(tokenize(prefix))
    verified = min(_lcp(prefix_ids, actual_ids),
                   *(_lcp(prefix_ids, other) for other in alternate_ids), token_cap)
    ceiling = min(verified, len(actual_ids) - 1) // PAGE_SIZE * PAGE_SIZE
    if ceiling < PAGE_SIZE:
        return {"admitted": False, "reason": "no_exact_shared_page"}
    return {"admitted": True, "reason": "exact_full_chat_token_prefix",
            "native_checkpoint_position_ceiling": ceiling,
            "verified_prefix_tokens": verified, "prefix_sha256": _sha(prefix.encode()),
            "actual_rendered_prompt_sha256": _sha(actual.encode()),
            "actual_token_ids_sha256": _sha(_encoded(actual_ids)),
            "full_actual_token_count": len(actual_ids)}

from dataclasses import dataclass


@dataclass(frozen=True)
class Ticket:
    tokens: object
    context: tuple
    boundary: int
    token_sha256: str


@dataclass
class SharedJob:
    boundary: int
    attempted: bool = False
    shared_plan: bool = False


class B2SharedPrefix:
    """Two bounded admission tickets; completed membership owns no extra Pair."""
    def __init__(self, coordinator, token_cap, template, tokenizer, renderer):
        if (type(token_cap) is not int or not 256 <= token_cap <= 393216 or token_cap % 256
                or coordinator.pairs.max_bytes != 1024**3 or coordinator.pairs.max_checkpoints != 8
                or coordinator.native.max_size != 4*1024**3
                or coordinator.pairs._all() or coordinator.pairs.jobs):
            raise ValueError('B2 shared requires fresh exact1GiB/eight-pair/native4GiB owner')
        self.owner=coordinator;self.token_cap=token_cap
        # Epoch: None or the exact target tier attached before this object.
        # Shared checkpoints hold no page references; validity stays is_resumable.
        self.tier=coordinator.tier
        self.context=(template,tokenizer,renderer);self.prepared=None;self.tickets={};self.completed=set()
        g=coordinator.generator
        self.owners=(g.model,g.cache,g.draft_cache,g.recurrent_cache)
        self.owner_serial=getattr(g.cache,'owner_serial',None)

    def fence(self):
        self.owner.check_owner()
        g=self.owner.generator
        if (any(a is not b for a,b in zip(self.owners,(g.model,g.cache,g.draft_cache,g.recurrent_cache)))
                or getattr(g.cache,'owner_serial',None)!=self.owner_serial or self.owner.tier is not self.tier):
            self.owner._terminal(RuntimeError('B2 shared ownership epoch or target tier changed'))

    @contextmanager
    def prepared_chat(self, ids, messages, tools, effort, prompt, *, template, tokenizer, renderer):
        import torch
        self.fence()
        if self.prepared is not None or any(a is not b for a,b in zip(self.context,(template,tokenizer,renderer))):
            raise ValueError('One exact bound API render scope required')
        if ids.device.type!='cpu' or ids.ndim!=2 or ids.shape[0]!=1 or ids.dtype!=torch.long:
            raise ValueError('Actual API CPU int64 token tensor required')
        result=derive_boundary(messages,tools,effort,prompt,ids[0].tolist(),
            lambda m,e,t:renderer(template,m,e,t),
            lambda text:tokenizer.encode(text,encode_special_tokens=True)[0].tolist(),self.token_cap)
        ticket=None if not result['admitted'] else Ticket(ids,self.context,
            result['native_checkpoint_position_ceiling'],result['actual_token_ids_sha256'])
        scope=[ticket,None];self.prepared=scope
        try:yield
        except BaseException:
            if scope[1] is not None:self.drop_ticket(scope[1])
            raise
        finally:self.prepared=None

    def bind_ticket(self, job):
        """Synchronous AsyncJob enqueue seam, before delayed register task exists."""
        import torch
        self.fence();scope=self.prepared
        if scope is None or scope[0] is None:return
        if scope[1] is not None or job in self.tickets or job in self.owner.pairs.jobs:
            raise ValueError('A shared API ticket can bind exactly one fresh job')
        if len(self.tickets)+len(self.owner.pairs.jobs)>=2 or len(job.sequences)!=1:
            raise ValueError('Only two shared pending/active job admissions')
        ticket=scope[0]
        if not torch.equal(job.sequences[0].input_ids.torch(),ticket.tokens):
            raise ValueError('Constructed native job differs from same API token tensor')
        self.tickets[job]=ticket;scope[1]=job

    def drop_ticket(self,job):self.tickets.pop(job,None)

    def prepare(self,job,state):
        import torch
        ticket=self.tickets.pop(job,None)
        if ticket is None:return
        self.fence();seq=job.sequences[0];actual=seq.sequence_ids.torch()
        if (ticket.context is not self.context or not torch.equal(seq.input_ids.torch(),ticket.tokens)
                or not torch.equal(actual,ticket.tokens) or _sha(_encoded(actual[0].tolist()))!=ticket.token_sha256):
            raise ValueError('Delayed native prepare lost its exact admitted token/renderer epoch')
        self.prune();boundary=ticket.boundary;tokens=actual[0].tolist();common=0
        for cp in self.owner.pairs.completed.values():
            if cp.identity in self.completed and self.owner.valid(cp):
                common=max(common,min(boundary,_lcp(tokens,cp.prefix[0].tolist()))//256*256)
        if common:boundary=common;self.owner.metrics['shared_adapted_boundaries']+=1
        state.shared_info=SharedJob(boundary)
        for cp in sorted(self.owner.pairs.completed.values(),key=lambda p:p.position,reverse=True):
            if cp.position<=boundary and self.owner.valid(cp) and torch.equal(actual[:,:cp.position],cp.prefix):
                state.shared=cp;break
        cp=state.plan
        state.shared_info.shared_plan=cp is not None and cp.identity in self.completed and cp.position<=boundary

    def restored(self,job,cp):
        state=self.owner.pairs.jobs[job];info=state.shared_info
        if info is not None and info.shared_plan and cp.identity in self.completed:
            self.owner.metrics['shared_hits']+=1;self.owner.metrics['shared_restored_tokens']+=cp.position

    def prefill(self,job,results,native,*args,**kwargs):
        self.fence();state=self.owner.pairs.jobs[job];info=state.shared_info
        if info is None or info.attempted:return native(job,results,*args,**kwargs)
        if len(args)>1 or set(kwargs)-{'chunk_size'} or args and 'chunk_size' in kwargs:
            raise TypeError('Exact native optional chunk_size once')
        chunk=args[0] if args else kwargs.get('chunk_size')
        if chunk is not None and (type(chunk) is not int or chunk<=0):raise ValueError('Positive per-call fair chunk')
        pos=job.sequences[0].kv_position;boundary=info.boundary
        if pos>=boundary:
            info.attempted=True;self.owner.metrics['shared_boundary_overshoots']+=int(pos>boundary)
            return native(job,results,*args,**kwargs)
        cap=min(self.owner.generator.max_chunk_size if chunk is None else chunk,boundary-pos)
        result=native(job,results,chunk_size=cap)
        self.fence();pos=job.sequences[0].kv_position
        if pos>=boundary:
            info.attempted=True
            if pos==boundary:
                # Use the same hooked stash/reservation/raw-window path. No rewind.
                job.maybe_stash_recurrent(self.owner.native,PAGE_SIZE)
                candidates=state.images()
                cp=next((p for p in candidates if p.position==boundary and self.owner.valid(p)),None)
                if cp is not None:
                    self.owner.pairs.adopt_shared(job,cp);self.owner.metrics['shared_captures']+=1
                else:self.owner.metrics['shared_capture_skips']+=1
            else:self.owner.metrics['shared_boundary_overshoots']+=1
        return result

    def finish(self,cp,success):
        if success and cp is not None and self.owner.pairs.completed.get(cp.identity) is cp:self.completed.add(cp.identity)
        self.prune()

    def prune(self):self.completed.intersection_update(self.owner.pairs.completed)

    def close(self):
        self.tickets.clear();self.completed.clear();self.prepared=None;self.context=None
