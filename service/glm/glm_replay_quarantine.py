"""Recurrent replay quarantine, installed by the service in the parent process (P16).

Engine patch 0004(b) (public-repo, "GDN replay quarantine") is not carried onto upstream dev 0f685aa9fd; this module
installs the same contract from the service, on the hash-pinned engine classes:

- GDNState.replay_failed is False until a settling rewind fails;
- a rewind that dispatches a replay (last_history > 0) and fails marks the state and raises RecurrentReplayError:
  under tensor parallelism some ranks may have replayed and others not, so the slot's rows are of uncertain content;
- stash() and free() refuse a quarantined state (no checkpoint of it, no recycling of its slot);
- Generator.reap_failed_job re-raises an error marked recurrent_replay_uncertain instead of recycling the job's pages
  and slot, so the failure reaches the async error sink (and the B2 owner latches the shared generator).

PairedState (upstream 7de419b) wraps each job's target state; its rewind/stash/free delegate to the GDNState, so the
wrappers below see every path. Importing this module does not load Torch or the engine.
"""


class RecurrentReplayError(RuntimeError):
    """Replay execution or staging custody is uncertain; do not reuse this state."""
    recurrent_replay_uncertain = True


def install():
    """Wrap GDNState.rewind/stash/free and Generator.reap_failed_job once; idempotent."""
    from exllamav3.modules import gated_delta_net
    from exllamav3.generator.generator import Generator
    GDNState = gated_delta_net.GDNState
    rewind, stash, free = GDNState.rewind, GDNState.stash, GDNState.free
    reap = Generator.reap_failed_job
    if getattr(rewind, "_glm_replay_quarantine", False):
        if not all(getattr(f, "_glm_replay_quarantine", False) for f in (stash, free, reap)):
            raise ValueError("Replay quarantine is only partly installed")
        return
    if any(getattr(f, "_glm_replay_quarantine", False) for f in (stash, free, reap)):
        raise ValueError("Replay quarantine is only partly installed")
    if hasattr(gated_delta_net, "RecurrentReplayError") or hasattr(GDNState, "replay_failed"):
        # An engine that still carries 0004(b) would double-wrap: refuse rather than guess which one owns the state.
        raise ValueError("The engine already carries a recurrent replay quarantine (patch 0004(b))")

    def guarded_rewind(self, num_tokens):
        if self.replay_failed:
            raise RecurrentReplayError("Recurrent state is quarantined after a replay failure")
        if self.last_history <= 0 or num_tokens < 0 or num_tokens > self.last_history:
            # No replay is dispatched: a position correction, or upstream's own argument assertion.
            return rewind(self, num_tokens)
        try:
            return rewind(self, num_tokens)
        except BaseException as error:
            # The replay dispatch is the only fallible step after the assertions: settlement is uncertain.
            self.replay_failed = True
            if isinstance(error, Exception):
                raise RecurrentReplayError(
                    "Recurrent replay failed before settlement; rank custody is uncertain") from error
            raise

    def guarded_stash(self):
        if self.replay_failed:
            raise RecurrentReplayError("Cannot stash a quarantined recurrent state")
        return stash(self)

    def guarded_free(self):
        if self.replay_failed:
            raise RecurrentReplayError("Cannot recycle a quarantined recurrent slot")
        return free(self)

    def guarded_reap(self, job, error, results):
        if getattr(error, "recurrent_replay_uncertain", False):
            raise error  # Shared target custody is uncertain; no per-job slot recycling.
        return reap(self, job, error, results)

    for function in (guarded_rewind, guarded_stash, guarded_free, guarded_reap):
        function._glm_replay_quarantine = True
    GDNState.replay_failed = False
    GDNState.rewind, GDNState.stash, GDNState.free = guarded_rewind, guarded_stash, guarded_free
    Generator.reap_failed_job = guarded_reap
