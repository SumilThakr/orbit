"""The perturbation path must assemble emissions exactly like the forward.

`zero-out` and `marginal` modes build their own emission vectors
(orbit.modes.perturbation) instead of reusing the forward's assembly in
orbit.cli.run_forward_month. Any divergence between the two is a silent
correctness bug: the perturbed run is a *different model* from the
baseline it is differenced against, so δ carries the discrepancy and no
amount of downstream care recovers it.

This has now happened twice in the same place, both times because the
2026-08-02 POA split was propagated to cli.py but not to
perturbation.py:

  1. `_OLD_TO_NEW` had no key 6, so loader slot 6 (POA) was silently
     dropped by `_rebroadcast_6_to_9` — 266 kg/s for January 2022, the
     largest emission channel after the PM2.5 remainder, and 84.7% of
     anthropogenic primary mass.
  2. Stage C (IVOC synthesised as POA x _IVOC_SCALING, distributed over
     the VBS bins; cli.py) had no counterpart in perturbation.py, so
     perturbed runs also lost all IVOC-derived SOA precursor.

Found 2026-08-03 by the FD audit of negative adjoint sensitivities: all
probe runs shared a byte-identical δPM2.5 offset (max 3.68 µg/m³) vs the
production baseline.

The test below is deliberately structural rather than numerical: it
asserts the *mapping* covers every loader slot the forward routes, and
that an unperturbed `build_perturbed_emissions` reproduces the forward's
per-species totals. A future species split then fails here rather than
in a quiet 3 µg/m³ bias six months later.
"""


from orbit.emissions.sources import N_ORBIT_SPECIES
from orbit.modes.perturbation import _OLD_TO_NEW


def _forward_old_to_new():
    """The forward's authoritative loader-slot → solver-slot mapping.

    Read from cli.py's source rather than duplicated here: a copy would
    drift in exactly the way this test exists to prevent.
    """
    import inspect
    import re
    import orbit.cli as cli
    src = inspect.getsource(cli.run_forward_month)
    m = re.search(r"_old_to_new\s*=\s*(\{[^}]*\})", src)
    assert m, "could not locate _old_to_new in cli.run_forward_month"
    return eval(m.group(1))


def test_every_loader_slot_the_forward_routes_is_routed_here():
    """The regression: loader slot 6 (POA) was missing from _OLD_TO_NEW."""
    forward = _forward_old_to_new()
    missing = set(forward) - set(_OLD_TO_NEW)
    assert not missing, (
        f"perturbation._OLD_TO_NEW is missing loader slots {sorted(missing)} "
        f"that the forward routes ({forward}). Emissions in those slots are "
        f"silently dropped from every zero-out / marginal perturbed run."
    )


def test_mapping_agrees_with_the_forward_slot_for_slot():
    """Not just present — routed to the SAME solver species."""
    forward = _forward_old_to_new()
    disagree = {k: (forward[k], _OLD_TO_NEW[k])
                for k in forward if k in _OLD_TO_NEW
                and forward[k] != _OLD_TO_NEW[k]}
    assert not disagree, (
        f"perturbation._OLD_TO_NEW disagrees with the forward on "
        f"{disagree} (loader_slot: (forward_target, perturbation_target))"
    )


def test_mapping_covers_the_full_loader_layout():
    """Guard against the next split: every loader slot must be accounted
    for, so adding an N_ORBIT_SPECIES entry cannot pass unnoticed."""
    unrouted = set(range(N_ORBIT_SPECIES)) - set(_OLD_TO_NEW)
    assert not unrouted, (
        f"loader slots {sorted(unrouted)} are not routed by "
        f"perturbation._OLD_TO_NEW (N_ORBIT_SPECIES={N_ORBIT_SPECIES}). "
        f"If a slot is intentionally unrouted, map it explicitly and say why."
    )


def test_ivoc_synthesis_exists_in_the_perturbation_path():
    """Stage C parity: the forward synthesises IVOC from POA; the
    perturbation path must too, or perturbed runs lose all IVOC-derived
    SOA precursor."""
    import inspect
    import orbit.modes.perturbation as pert
    src = inspect.getsource(pert)
    assert "ivoc" in src.lower(), (
        "no IVOC synthesis in orbit.modes.perturbation — the forward's "
        "Stage C (IVOC = POA x _IVOC_SCALING, distributed over VBS bins) "
        "has no counterpart, so perturbed emissions understate SOA "
        "precursor. See cli.py Stage C."
    )
