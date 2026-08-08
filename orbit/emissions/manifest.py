"""Emission manifest: a declarative description of the baseline inventory.

A manifest is a small YAML file listing *which* emission files make up a run
and how each should be interpreted -- units, stack parameters for plume rise,
VOC parent class, whether the file carries its own diurnal bins. It does not
contain emissions data and it does not fetch anything; the NetCDF files
themselves are located at run time under the emissions directory.

The default manifest shipped with ORBIT
(``orbit/data/emissions_sas_2022_poa.yaml``) reproduces the published South
Asia 2022 inventory. To run a different inventory, copy it, edit, and pass
``--emissions-manifest``.

Schema
------
::

    name: sas_2022
    description: free text

    defaults:                 # optional; applied to every source
      units: kg/m2/s
      time_index: month       # 'month' -> select month-1; or an integer; or null

    sources:
      - file: ceds_nox_anthro_2022_monthly_high.nc
        stack: {height: 220, diameter: 6, temperature: 410, velocity: 20}
      - file: ceds_voc_anthro_2022_monthly.nc
        voc_class: anthro
      - file: cams_soil_nox_climatology_diurnal.nc
        bin_axis: true
      - file: something_optional.nc
        required: false       # absence is reported, not fatal

``file`` is the only required key. A source whose file is absent is an error
unless it is marked ``required: false`` or the caller passes
``allow_missing=True`` -- a run with silently missing emissions would produce a
plausible-looking but wrong answer.
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from orbit.emissions.sources import EmissionSource

# Parent classes understood by the VBS yield tables. ``anthro`` and ``bio_voc``
# are meta-classes resolved per cell (NOx regime) or per split configuration.
VALID_VOC_CLASSES = {
    "anthro", "anthro_high_nox", "anthro_low_nox",
    "bio_voc", "bio_monoterpene", "bio_isoprene",
    "biomass_burning", "ivoc",
}

_STACK_KEYS = ("height", "diameter", "temperature", "velocity")


@dataclass
class ManifestEntry:
    """One emission file and how to interpret it."""

    file: str
    units: str = "kg/m2/s"
    time_index: Any = "month"
    voc_class: Optional[str] = None
    stack: Optional[Dict[str, float]] = None
    bin_axis: bool = False
    required: bool = True
    note: str = ""

    def resolve_time_index(self, month: int) -> Optional[int]:
        if self.time_index is None:
            return None
        if isinstance(self.time_index, str):
            if self.time_index.lower() != "month":
                raise ValueError(
                    f"time_index for {self.file!r} must be 'month', an "
                    f"integer, or null; got {self.time_index!r}"
                )
            return month - 1
        return int(self.time_index)

    def identity(self) -> Dict[str, Any]:
        """The fields that change what the emissions actually are.

        Used for the provenance hash, so that editing units or stack
        parameters invalidates a baseline even when filenames are unchanged.
        """
        return {
            "file": self.file,
            "units": self.units,
            "time_index": self.time_index,
            "voc_class": self.voc_class,
            "stack": self.stack,
            "bin_axis": self.bin_axis,
        }


@dataclass
class LoadReport:
    """What a manifest resolved to on disk, for logging and error messages."""

    loaded: List[str] = field(default_factory=list)
    missing_required: List[str] = field(default_factory=list)
    missing_optional: List[str] = field(default_factory=list)

    def summary(self) -> str:
        parts = [f"{len(self.loaded)} loaded"]
        if self.missing_optional:
            parts.append(f"{len(self.missing_optional)} optional missing")
        if self.missing_required:
            parts.append(f"{len(self.missing_required)} REQUIRED MISSING")
        return ", ".join(parts)


class MissingEmissionsError(RuntimeError):
    """Raised when required emission files are absent."""


@dataclass
class EmissionManifest:
    """A parsed manifest: the named list of sources that make up a run.

    Construct with :meth:`from_yaml`; resolve against an emissions
    directory for a given month with :meth:`build_sources`.
    :meth:`content_hash` gives a short provenance hash over every field
    that changes the emissions, so a baseline cannot be silently reused
    across manifests that interpret the same files differently.
    """

    name: str = ""
    description: str = ""
    entries: List[ManifestEntry] = field(default_factory=list)
    path: str = ""

    # ── construction ──────────────────────────────────────────────────────

    @classmethod
    def from_yaml(cls, path: str) -> "EmissionManifest":
        import yaml

        with open(path) as fh:
            data = yaml.safe_load(fh) or {}
        if not isinstance(data, dict):
            raise ValueError(
                f"Emission manifest must be a top-level mapping; got "
                f"{type(data).__name__} from {path}"
            )

        defaults = data.get("defaults") or {}
        raw_sources = data.get("sources")
        if not raw_sources:
            raise ValueError(f"Emission manifest {path} lists no sources")

        entries: List[ManifestEntry] = []
        seen = set()
        for i, raw in enumerate(raw_sources):
            if not isinstance(raw, dict):
                raise ValueError(
                    f"sources[{i}] in {path} must be a mapping; got {raw!r}"
                )
            if "file" not in raw:
                raise ValueError(f"sources[{i}] in {path} missing required 'file'")
            fname = str(raw["file"]).strip()
            if fname in seen:
                raise ValueError(f"Duplicate source {fname!r} in {path}")
            seen.add(fname)

            voc_class = raw.get("voc_class")
            if voc_class is not None:
                voc_class = str(voc_class)
                if voc_class not in VALID_VOC_CLASSES:
                    raise ValueError(
                        f"sources[{i}] ({fname}) in {path}: unknown voc_class "
                        f"{voc_class!r}. Valid: {sorted(VALID_VOC_CLASSES)}"
                    )

            stack = raw.get("stack")
            if stack is not None:
                if not isinstance(stack, dict):
                    raise ValueError(
                        f"sources[{i}] ({fname}) in {path}: 'stack' must be a "
                        f"mapping with keys {list(_STACK_KEYS)}"
                    )
                missing = [k for k in _STACK_KEYS if k not in stack]
                if missing:
                    raise ValueError(
                        f"sources[{i}] ({fname}) in {path}: 'stack' missing "
                        f"{missing}; all of {list(_STACK_KEYS)} are required"
                    )
                stack = {k: float(stack[k]) for k in _STACK_KEYS}

            entries.append(ManifestEntry(
                file=fname,
                units=str(raw.get("units", defaults.get("units", "kg/m2/s"))),
                time_index=raw.get("time_index", defaults.get("time_index", "month")),
                voc_class=voc_class,
                stack=stack,
                bin_axis=bool(raw.get("bin_axis", False)),
                required=bool(raw.get("required", True)),
                note=str(raw.get("note", "")),
            ))

        return cls(
            name=str(data.get("name", os.path.basename(path))),
            description=str(data.get("description", "")),
            entries=entries,
            path=os.path.abspath(path),
        )

    # ── use ───────────────────────────────────────────────────────────────

    def build_sources(self, emission_dir: str, month: int,
                      allow_missing: bool = False):
        """Resolve the manifest against ``emission_dir`` for one month.

        Returns ``(sources, report)``. Raises :class:`MissingEmissionsError` if
        a required file is absent and ``allow_missing`` is False.
        """
        sources: List[EmissionSource] = []
        report = LoadReport()

        for entry in self.entries:
            path = os.path.join(emission_dir, entry.file)
            if not os.path.exists(path):
                if entry.required:
                    report.missing_required.append(entry.file)
                else:
                    report.missing_optional.append(entry.file)
                continue

            kwargs: Dict[str, Any] = dict(
                path=path,
                format="netcdf",
                units=entry.units,
                time_index=entry.resolve_time_index(month),
            )
            if entry.bin_axis:
                kwargs["bin_axis"] = True
            if entry.stack:
                kwargs.update(
                    elevated=True,
                    stack_height=entry.stack["height"],
                    stack_diam=entry.stack["diameter"],
                    stack_temp=entry.stack["temperature"],
                    stack_vel=entry.stack["velocity"],
                )
            if entry.voc_class:
                kwargs["voc_parent_class"] = entry.voc_class

            sources.append(EmissionSource(**kwargs))
            report.loaded.append(entry.file)

        if report.missing_required and not allow_missing:
            raise MissingEmissionsError(
                f"{len(report.missing_required)} required emission file(s) not "
                f"found under {emission_dir}:\n  "
                + "\n  ".join(report.missing_required)
                + f"\n\nManifest: {self.path or self.name}\n"
                "Fix the emissions directory (ORBIT_EMISSION_DIR), mark these "
                "sources 'required: false' in the manifest, or pass "
                "--allow-missing-emissions to proceed without them. Running "
                "with emissions silently absent would give a wrong answer."
            )

        return sources, report

    def voc_classes(self) -> Dict[str, str]:
        """Basename -> VOC parent class, for the VBS distributor."""
        return {e.file: e.voc_class for e in self.entries if e.voc_class}

    def content_hash(self) -> str:
        """Short hash over every field that changes the emissions.

        Covers units, stack parameters, VOC class and bin axis -- not just
        filenames -- so a baseline cannot be reused across manifests that
        differ only in how the same files are interpreted.
        """
        payload = json.dumps(
            [e.identity() for e in sorted(self.entries, key=lambda x: x.file)],
            sort_keys=True,
        )
        return hashlib.sha256(payload.encode()).hexdigest()[:16]

    def describe(self, report: Optional[LoadReport] = None) -> str:
        """Human-readable inventory listing for the run log."""
        lines = [f"Emissions manifest: {self.name}"]
        if self.path:
            lines.append(f"  file: {self.path}")
        lines.append(f"  hash: {self.content_hash()}")
        if report is not None:
            lines.append(f"  {report.summary()}")
            for f in report.loaded:
                entry = next(e for e in self.entries if e.file == f)
                tags = []
                if entry.voc_class:
                    tags.append(f"voc={entry.voc_class}")
                if entry.stack:
                    tags.append(f"stack={entry.stack['height']:g}m")
                if entry.bin_axis:
                    tags.append("per-bin")
                suffix = ("  [" + ", ".join(tags) + "]") if tags else ""
                lines.append(f"    + {f}{suffix}")
            for f in report.missing_optional:
                lines.append(f"    - {f}  [optional, absent]")
            for f in report.missing_required:
                lines.append(f"    ! {f}  [REQUIRED, ABSENT]")
        return "\n".join(lines)


#: Seconds in a 365-day year, for annualising a monthly emission rate.
_SECONDS_PER_YEAR = 365.0 * 24.0 * 3600.0


def source_mass_budget(sources, grid, indexer, verbose=False):
    """Per-source emitted mass, annualised.

    Returns ``(rows, totals)`` where each row is
    ``(basename, species_label, tg_per_year, tags)`` and ``totals`` maps a
    species label to Tg/yr.

    The value is **this month's emission rate scaled to a full year**, not the
    calendar-year total: monthly inventory files hold a rate, and annualising
    gives a number comparable with published inventories. A month with unusual
    biomass burning will therefore annualise high, which is the intended
    reading -- it says "at January's rate, this much per year".

    A mass budget is the cheapest way to catch the errors a file listing
    cannot show: a wrong ``units`` declaration, a bad regrid, or a truncated
    inventory are all invisible as filenames and obvious as a total.
    """

    from orbit.emissions.sources import N_ORBIT_SPECIES

    labels = ["SOA/VOC", "PrimaryPM2.5", "TotalNH", "SO2", "pSO4", "TotalNO3"]
    rows, totals = [], {}
    vol = grid.volume.ravel()

    for src in sources:
        try:
            if src.format in ("shapefile", "geopackage"):
                from orbit.emissions.shapefile import load_shapefile_source
                emis = load_shapefile_source(src, grid, False)
            else:
                from orbit.emissions.netcdf import load_netcdf_source
                emis = load_netcdf_source(src, grid, False)
        except Exception as exc:            # never let a diagnostic kill a run
            rows.append((os.path.basename(src.path), "?", float("nan"),
                         f"unreadable: {exc}"))
            continue
        # emis is (N_ORBIT_SPECIES, nz, ny, nx) in ug/m3/s; mass rate is the
        # volume-weighted sum, converted ug/s -> Tg/s -> Tg/yr.
        flat = emis.reshape(N_ORBIT_SPECIES, -1)
        per_species_kg_s = (flat * vol[None, :]).sum(axis=1) * 1e-9
        for i, kg_s in enumerate(per_species_kg_s):
            if kg_s <= 0:
                continue
            tg_yr = kg_s * _SECONDS_PER_YEAR * 1e-9
            label = labels[i] if i < len(labels) else f"idx{i}"
            tags = []
            if src.voc_parent_class:
                tags.append(f"voc={src.voc_parent_class}")
            if getattr(src, "elevated", False) and src.stack_height:
                tags.append(f"stack={src.stack_height:g}m")
            rows.append((os.path.basename(src.path), label, tg_yr,
                         ", ".join(tags)))
            totals[label] = totals.get(label, 0.0) + tg_yr
    return rows, totals


def format_mass_budget(rows, totals) -> str:
    """Render the mass budget as a log table."""
    out = ["  source                                        species        Tg/yr  tags",
           "  " + "-" * 74]
    for name, label, tg, tags in sorted(rows, key=lambda r: (-r[2] if r[2] == r[2] else 0)):
        out.append(f"  {name:<44s} {label:<12s} {tg:>9.4f}  {tags}")
    out.append("  " + "-" * 74)
    out.append("  TOTAL  " + " | ".join(
        f"{k} {v:.3f}" for k, v in sorted(totals.items(), key=lambda kv: -kv[1])))
    out.append("  (Tg/yr = this month's emission rate scaled to a full year,")
    out.append("   not the calendar-year total.)")
    return "\n".join(out)


def default_manifest_path() -> str:
    """Path to the manifest shipped with ORBIT.

    The POA-split manifest is the configuration the published South Asia
    2022 baseline was produced with, and the marginal mode's baseline-hash
    check requires the active manifest to match the baseline being
    linearised around.
    """
    return os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "data", "emissions_sas_2022_poa.yaml",
    )


def load_manifest(path: Optional[str] = None) -> EmissionManifest:
    """Load a manifest, falling back to the shipped default.

    Resolution order: explicit ``path`` argument, then
    ``ORBIT_EMISSION_MANIFEST``, then the shipped default.
    """
    chosen = path or os.environ.get("ORBIT_EMISSION_MANIFEST") or default_manifest_path()
    if not os.path.exists(chosen):
        raise FileNotFoundError(f"Emission manifest not found: {chosen}")
    return EmissionManifest.from_yaml(chosen)
