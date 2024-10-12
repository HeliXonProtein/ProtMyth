# Copyright (c) 2024 Helixon Limited.
#
# This file is a part of ProtMyth and is released under the MIT License.
# Thanks for using ProtMyth!

"""Constants for protmyth data.
"""

STANDARD_DNA = [
    "A",
    'T',
    "C",
    'G',
]

#   B == 5-bromouridine
#   D == 5,6-dihydrouridine
#   S == thiouridine
#   W == wyosine
AUGMENTED_DNA = STANDARD_DNA + [
    "B",
    "D",
    "S",
    "W",
]

#  follow https://www.cottongen.org/help/nomenclature/IUPAC_nt
#   B in AMBIGUOUS_DNA is 'NOT A' (TCG, letter B comes after A)
#   This is different from the AUGMENTED_DNA definition.
AMBIGUOUS_DNA = [
    "R",
    "Y",
    "W",
    "S",
    "M",
    "K",
    "H",
    "B",
    "V",
    "D",
    "N",
]

# "X" is included in the following _VALUES and _COMPLEMENT dictionaries,
# for historical reasons although it is not an IUPAC nucleotide.
AMBIGUOUS_DNA_VALUES = {
    "M": "AC",
    "R": "AG",
    "W": "AT",
    "S": "CG",
    "Y": "CT",
    "K": "GT",
    "V": "ACG",
    "H": "ACT",
    "D": "AGT",
    "B": "CGT",
    "X": "GATC",
    "N": "GATC",
}

AMBIGUOUS_DNA_COMPLEMENT = {
    "A": "T",
    "C": "G",
    "G": "C",
    "T": "A",
    "M": "K",
    "R": "Y",
    "W": "W",
    "S": "S",
    "Y": "R",
    "K": "M",
    "V": "B",
    "H": "D",
    "D": "H",
    "B": "V",
    "X": "X",
    "N": "N",
}

STANDARD_RNA = [
    "A",
    'U',
    "C",
    'G',
]

AUGMENTED_RNA = STANDARD_RNA + [
    "B",
    "D",
    "S",
    "W",
]

AMBIGUOUS_RNA = [
    "R",
    "Y",
    "W",
    "S",
    "M",
    "K",
    "H",
    "B",
    "V",
    "D",
    "N",
]

AMBIGUOUS_RNA_VALUES = {
    "M": "AC",
    "R": "AG",
    "W": "AU",
    "S": "CG",
    "Y": "CU",
    "K": "GU",
    "V": "ACG",
    "H": "ACU",
    "D": "AGU",
    "B": "CGU",
    "X": "GAUC",
    "N": "GAUC",
}

AMBIGUOUS_DNA_COMPLEMENT = {
    "A": "U",
    "C": "G",
    "G": "C",
    "U": "A",
    "M": "K",
    "R": "Y",
    "W": "W",
    "S": "S",
    "Y": "R",
    "K": "M",
    "V": "B",
    "H": "D",
    "D": "H",
    "B": "V",
    "X": "X",
    "N": "N",
}
