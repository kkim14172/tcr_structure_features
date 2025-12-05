#!/bin/bash
set -euo pipefail

ROSETTA3="/Applications/rosetta.binary.m1.release-408/main/source/bin"

if [[ $# -lt 2 ]]; then
  echo "Usage: $0 <pdb_basename_without_ext> <output_dir>"
  exit 1
fi

PDB_DIR="$1"
OUTDIR="$2"
ABS_PDB_DIR=$(cd "$(dirname "$PDB_DIR")"; pwd)/"$(basename "$PDB_DIR")"

if [[ ! -d "$ABS_PDB_DIR" ]]; then
  echo "Error: PDB directory '$ABS_PDB_DIR' not found."
  exit 1
fi
cd "$ABS_PDB_DIR"

mkdir -p "$OUTDIR"

# Make sure there are actually pdb files
shopt -s nullglob
pdbs=( *.pdb )
if (( ${#pdbs[@]} == 0 )); then
  echo "Error: no .pdb files found in '$ABS_PDB_DIR'"
  exit 1
fi

"${ROSETTA3}/InterfaceAnalyzer.macosclangrelease" \
  -s "${pdbs[@]}" \
  -interface A_B \
  -pack_input true \
  -pack_separated true \
  -compute_packstat true \
  -tracer_data_print false \
  -atomic_burial_cutoff 0.01 \
  -sasa_calculator_probe_radius 1.4 \
  -pose_metrics::interface_cutoff 8 \
  -out:file:score_only "${OUTDIR}/interface_analyzer.out" \
  -use_jobname true \
  -use_input_sc \
  -add_regular_scores_to_scorefile true \
  -overwrite


"${ROSETTA3}/residue_energy_breakdown.macosclangrelease" \
    -s "${pdbs[@]}" \
    -out:file:silent "${OUTDIR}/residue_energy_breakdown.out" \
    -overwrite