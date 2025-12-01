import subprocess
import re
import ast
from typing import Dict, Any, List, Tuple

ID_TABLE_HDR = re.compile(r"^\s*ID\s+name\s*$")
ID_TABLE_ROW = re.compile(r"^\s*(\d+)\s+([A-Za-z0-9_ -]+)\s*$")
N_PROT_CHAINS = re.compile(r"^\s*Number of protein chains:\s*(\d+)\s*$")
ARRAY_LINE = re.compile(r"^array\((\[.*?\])")
SCHEME_LINE = re.compile(r"^\s*Scheme\s*=\s*([A-Za-z0-9_+-]+)\s*$")
SECTION_HDR = re.compile(r"^\s*#\s*(.*)\s*$")
BAR_TABLE_ROW = re.compile(r"^\s*\|(.*?)\|\s*$")
RES_LINE = re.compile(r"^\s*([A-Za-z])\s+(\d+)\s+([A-Z-])\s*$")
END_BLOCK = re.compile(r"^\s*//\s*$")

def _parse_bracket_list(s: str) -> List[str]:
    """
    Extract the [...] part and ast.literal_eval safely, after stripping quotes.
    """
    m = ARRAY_LINE.search(s)
    if not m:
        return []
    raw = m.group(1)
    # Ensure it’s a clean Python list literal
    return ast.literal_eval(raw)

def _parse_bar_table(lines: List[str], start_i: int) -> Tuple[List[str], List[List[str]], int]:
    """
    Parse a bar-delimited table starting at start_i (header + one or more rows).
    Returns (headers, rows, next_index_after_table)
    """
    headers, rows = [], []
    i = start_i
    # header
    mh = BAR_TABLE_ROW.match(lines[i])
    if not mh:
        return headers, rows, i
    headers = [c.strip() for c in mh.group(1).split("|") if c.strip()]
    i += 1
    # data rows
    while i < len(lines):
        m = BAR_TABLE_ROW.match(lines[i])
        if not m:
            break
        cells = [c.strip() for c in m.group(1).split("|") if c.strip()]
        # Only accept rows that look like the header length
        if cells and (len(cells) == len(headers)):
            rows.append(cells)
            i += 1
        else:
            # Likely an empty separator or different section
            break
    return headers, rows, i

def parse_anarci_stdout(stdout: str) -> Dict[str, Any]:
    lines = stdout.splitlines()
    i = 0
    out: Dict[str, Any] = {
        "id_table": [],                  # list of {"ID": int, "name": str}
        "num_protein_chains": None,      # int
        "arrays": [],                    # each a Python list (from the array(...) lines)
        "anarci": {
            "scheme": None,              # e.g., "aho"
            "most_significant_hmm": {},  # dict of parsed columns
            "most_identical_germlines": [],  # list of dicts
            "numbered_sequence": []      # list of (int, AA) tuples
        }
    }

    # Pass 1: ID/name table and miscellany prior to ANARCI block
    while i < len(lines):
        line = lines[i]

        # ID / name table
        if ID_TABLE_HDR.match(line):
            i += 1
            while i < len(lines):
                mrow = ID_TABLE_ROW.match(lines[i])
                if not mrow:
                    break
                out["id_table"].append({"ID": int(mrow.group(1)), "name": mrow.group(2).strip()})
                i += 1
            continue

        # Number of protein chains
        m = N_PROT_CHAINS.match(line)
        if m:
            out["num_protein_chains"] = int(m.group(1))
            i += 1
            continue

        # array([...]) lines
        if ARRAY_LINE.search(line):
            out["arrays"].append(_parse_bracket_list(line))
            i += 1
            continue

        # Stop scanning early once we hit the ANARCI block marker
        if line.strip().startswith("# Input sequence") or line.strip().startswith("# ANARCI numbered"):
            break

        i += 1

    # Pass 2: ANARCI sections (from where we left off)
    # We will collect scheme, bar tables, and residue-numbered sequence until '//'
    while i < len(lines):
        line = lines[i]

        # Scheme
        msch = SCHEME_LINE.match(line)
        if msch:
            out["anarci"]["scheme"] = msch.group(1).strip().lower()
            i += 1
            continue

        # "Most significant HMM hit" table
        if line.strip().startswith("# Most significant HMM hit"):
            # Next two lines should be header and a row
            i += 1
            headers, rows, i = _parse_bar_table(lines, i)
            if rows:
                out["anarci"]["most_significant_hmm"] = dict(zip(headers, rows[0]))
            continue

        # "Most sequence-identical germlines" table
        if line.strip().startswith("# Most sequence-identical germlines"):
            i += 1
            headers, rows, i = _parse_bar_table(lines, i)
            out["anarci"]["most_identical_germlines"] = [dict(zip(headers, r)) for r in rows]
            continue

        # Numbered sequence lines like: "H 1     E"
        mres = RES_LINE.match(line)
        if mres:
            # chain = mres.group(1)  # 'H' etc., available if you need it
            idx = int(mres.group(2))
            aa = mres.group(3)
            out["anarci"]["numbered_sequence"].append((idx, aa))
            i += 1
            continue

        # End of ANARCI block (or sequence)
        if END_BLOCK.match(line):
            i += 1
            break

        i += 1

    return out

def run_anarci(sequence, scheme="a", assign_germline=True):
    
    command = ["bash", "/workspaces/tcr_structure_embedding/structure_feature/sequence/alignment/run_anarci.sh", 
            "-i", sequence.upper(), 
            "-s", scheme]
    
    if assign_germline is True: 
        command.extend("--assign_germline")
        
    result = subprocess.run(command, capture_output=True, text=True, check=True)
    if "# Most significant HMM hit" not in result.stdouot:
        return None
    
    parsed = parse_anarci_stdout(result.stdout)
    print(parsed["id_table"])
    print(parsed["num_protein_chains"])
    print(parsed["arrays"])  # three lists from the array(...) lines
    print(parsed["anarci"]["scheme"])
    print(parsed["anarci"]["most_significant_hmm"])
    print(parsed["anarci"]["most_identical_germlines"])
    print(parsed["anarci"]["numbered_sequence"][:10])  # first 10 tuples
    return parsed
    