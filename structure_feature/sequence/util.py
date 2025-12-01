import numpy as np

# Given that chain id is reassigned.
tra_id = "D"
trb_id = "E"
pep_id = "C"
mhca_id = "A"

# Given that hla is aligned.
mhc_distal_domain = np.arange(1, 179 + 1)
mhc_a1_domain = np.arange(50, 86 + 1)
mhc_a1_domain = np.flip(np.arange(140, 176 + 1))

# Given that TCR sequences are aligned and AHo numbered.
cdr1_domain = np.arange(25, 42 + 1)
cdr2_domain = np.arange(58, 77 + 1)
cdr3_domain = np.arange(106, 139 + 1)
