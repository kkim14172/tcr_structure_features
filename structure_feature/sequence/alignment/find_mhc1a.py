# import numpy as np
# import biotite.sequence as struc
# import biotite.sequence.align as align
# import biotite.structure.alphabet as strucalph

# def get_mhc1a_chain_loc(array, ref_mhc=None):

#     if ref_mhc is None:
#         import biotite.structure.io.pdb as pdb
#         pdb_file = pdb.PDBFile.read("/workspaces/tcr_structure_embedding/killing_assay/tcrmodel2_out/tcr7/ranked_0_proj.pdb")
#         atom_array = pdb_file.get_structure(model=1)
#         ref_mhc = atom_array[atom_array.chain_id == 'A']
    
#     structural_sequences = []
#     array = [ref_mhc] + array

#     sequences = []
#     for (i,s) in array:
#         seq, _ = struc.to_sequence(s)
#         sequences.append(seq[0])
        
#     for (i,s) in array:
#         pb_sequences, _ = strucalph.to_protein_blocks(s)
#         structural_sequences.append(pb_sequences[0].remove_undefined())
        
#     # Perform a multiple sequence alignment of the PB sequences
#     stc_alignment, _, _, _ = align.align_multiple(
#         structural_sequences, align.SubstitutionMatrix.std_protein_blocks_matrix(), gap_penalty=(-500, -100), terminal_penalty=False
#     )
#     seq_alignment, _, _, _ = align.align_multiple(
#         sequences, align.SubstitutionMatrix.std_protein_matrix(), gap_penalty=(-500, -100), terminal_penalty=False
#     )
#     alignment_score = align.get_pairwise_sequence_identity(seq_alignment)[0][1:] + align.get_pairwise_sequence_identity(stc_alignment)[0][1:]
#     return array[np.argmax(alignment_score)][0]


import numpy as np
import biotite.structure as struc
import biotite.sequence.align as align
import biotite.structure.alphabet as strucalph

def get_mhc1a_chain_id(array, ref_mhc=None):

    if ref_mhc is None:
        import biotite.structure.io.pdb as pdb
        pdb_file = pdb.PDBFile.read("/workspaces/tcr_structure_embedding/killing_assay/tcrmodel2_out/tcr7/ranked_0_proj.pdb")
        atom_array = pdb_file.get_structure(model=1)
        ref_mhc = atom_array[atom_array.chain_id == 'A']
    
    structural_sequences = []
    array = [ref_mhc] + array

    sequences = []
    for (i,s) in array:
        seq, _ = struc.to_sequence(s)
        sequences.append(seq[0])
        
    for (i,s) in array:
        pb_sequences, _ = strucalph.to_protein_blocks(s)
        structural_sequences.append(pb_sequences[0].remove_undefined())
        
    # Perform a multiple sequence alignment of the PB sequences
    stc_alignment, _, _, _ = align.align_multiple(
        structural_sequences, align.SubstitutionMatrix.std_protein_blocks_matrix(), gap_penalty=(-500, -100), terminal_penalty=False
    )
    seq_alignment, _, _, _ = align.align_multiple(
        sequences, align.SubstitutionMatrix.std_protein_matrix(), gap_penalty=(-500, -100), terminal_penalty=False
    )
    alignment_score = align.get_pairwise_sequence_identity(seq_alignment)[0][1:] + align.get_pairwise_sequence_identity(stc_alignment)[0][1:]
    return array[np.argmax(alignment_score)][0]