import biotite.structure as struc
import biotite.structure.io.pdbx as pdbx
import biotite.database.rcsb as rcsb

def retrieve_and_process_biological_assembly(PDB_ID, assembly_idx = 1):
    pdbx_file = pdbx.BinaryCIFFile.read(rcsb.fetch(PDB_ID, "bcif"))

    assemblies = pdbx.list_assemblies(pdbx_file)
    print("ID    name")
    print()
    for assembly_id, name in assemblies.items():
        print(f"{assembly_id:2}    {name}")
        
    assembly = pdbx.get_assembly(
        pdbx_file,
        assembly_id=str(assembly_idx),
        model=1,
        # To identify later which atoms belong to which protein type
        extra_fields=["label_entity_id"],
    )

    print("Number of protein chains:", struc.get_chain_count(assembly))

    mask = struc.filter_canonical_amino_acids(assembly) & struc.filter_polymer(assembly)
    mask2 = (~struc.filter_solvent(assembly)) & (~struc.filter_monoatomic_ions(assembly))
    filtered = assembly[mask & mask2]

    print("Number of protein chains after filtering:", struc.get_chain_count(filtered))
    return filtered