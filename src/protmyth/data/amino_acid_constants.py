# Copyright (c) 2024 Helixon Limited.
#
# This file is a part of ProtMyth and is released under the MIT License.
# Thanks for using ProtMyth!

"""Constants for protmyth data.
"""

import torch

from protmyth.data.types import ProteinSequenceDomain


# This is the standard residue order when coding AA type as a number.
STANDARD_AMINO_ACIDS_20 = [
    "A",
    "R",
    "N",
    "D",
    "C",
    "Q",
    "E",
    "G",
    "H",
    "I",
    "L",
    "K",
    "M",
    "F",
    "P",
    "S",
    "T",
    "W",
    "Y",
    "V",
]

#   from https://github.com/biopython/biopython/blob/5ee5e69e649dbe17baefe3919e56e60b54f8e08f/Bio/Data/IUPACData.py
#   B = "Asx";  aspartic acid or asparagine (D or N)
#   X = "Xxx";  unknown or 'other' amino acid
#   Z = "Glx";  glutamic acid or glutamine (E or Q)
#   http://www.chem.qmul.ac.uk/iupac/AminoAcid/A2021.html#AA212
#
#   J = "Xle";  leucine or isoleucine (L or I, used in NMR)
#   Mentioned in http://www.chem.qmul.ac.uk/iubmb/newsletter/1999/item3.html
#   Also the International Nucleotide Sequence Database Collaboration (INSDC)
#   (i.e. GenBank, EMBL, DDBJ) adopted this in 2006
#   http://www.ddbj.nig.ac.jp/insdc/icm2006-e.html
#
#   Xle (J); Leucine or Isoleucine
#   The residue abbreviations, Xle (the three-letter abbreviation) and J
#   (the one-letter abbreviation) are reserved for the case that cannot
#   experimentally distinguish leucine from isoleucine.
#
#   U = "Sec";  selenocysteine
#   http://www.chem.qmul.ac.uk/iubmb/newsletter/1999/item3.html
#
#   O = "Pyl";  pyrrolysine
#   http://www.chem.qmul.ac.uk/iubmb/newsletter/2009.html#item35

STANDARD_AMINO_ACIDS_22 = STANDARD_AMINO_ACIDS_20 + [
    "O",  # L-Pyrrolysine
    "U",  # L-Selenocysteine
]

IUPAC_AMINO_ACIDS = STANDARD_AMINO_ACIDS_22 + [
    "B",  # Asx; Asparagine or aspartic acid
    "Z",  # Glx; Glutamine or glutamic acid
    "J",  # Xle; Leucine or Isoleucine
]

STANDARD_AA_1to3 = {
    "A": "ALA",
    "R": "ARG",
    "N": "ASN",
    "D": "ASP",
    "C": "CYS",
    "Q": "GLN",
    "E": "GLU",
    "G": "GLY",
    "H": "HIS",
    "I": "ILE",
    "L": "LEU",
    "K": "LYS",
    "M": "MET",
    "F": "PHE",
    "P": "PRO",
    "S": "SER",
    "T": "THR",
    "W": "TRP",
    "Y": "TYR",
    "V": "VAL",
    "O": "PYL",
    "U": "SEC",
    "X": "UNK",
}

AUGMENTED_AA_1to3 = dict(
    list(STANDARD_AA_1to3.items()) + list({
        "B": "ASX",
        "Z": "GLX",
        "J": "XLE",
        "U": "SEC",
        "O": "PYL",
    }.items())
)

STANDARD_AA_3to1 = {v: k for k, v in STANDARD_AA_1to3.items()}
AUGMENTED_AA_3to1 = {v: k for k, v in AUGMENTED_AA_1to3.items()}

NON_STANDARD_SUBSTITUTIONS = {
    "2AS": "ASP",
    "3AH": "HIS",
    "5HP": "GLU",
    "ACL": "ARG",
    "AGM": "ARG",
    "AIB": "ALA",
    "ALM": "ALA",
    "ALO": "THR",
    "ALY": "LYS",
    "ARM": "ARG",
    "ASA": "ASP",
    "ASB": "ASP",
    "ASK": "ASP",
    "ASL": "ASP",
    "ASQ": "ASP",
    "AYA": "ALA",
    "BCS": "CYS",
    "BHD": "ASP",
    "BMT": "THR",
    "BNN": "ALA",
    "BUC": "CYS",
    "BUG": "LEU",
    "C5C": "CYS",
    "C6C": "CYS",
    "CAS": "CYS",
    "CCS": "CYS",
    "CEA": "CYS",
    "CGU": "GLU",
    "CHG": "ALA",
    "CLE": "LEU",
    "CME": "CYS",
    "CSD": "ALA",
    "CSO": "CYS",
    "CSP": "CYS",
    "CSS": "CYS",
    "CSW": "CYS",
    "CSX": "CYS",
    "CXM": "MET",
    "CY1": "CYS",
    "CY3": "CYS",
    "CYG": "CYS",
    "CYM": "CYS",
    "CYQ": "CYS",
    "DAH": "PHE",
    "DAL": "ALA",
    "DAR": "ARG",
    "DAS": "ASP",
    "DCY": "CYS",
    "DGL": "GLU",
    "DGN": "GLN",
    "DHA": "ALA",
    "DHI": "HIS",
    "DIL": "ILE",
    "DIV": "VAL",
    "DLE": "LEU",
    "DLY": "LYS",
    "DNP": "ALA",
    "DPN": "PHE",
    "DPR": "PRO",
    "DSN": "SER",
    "DSP": "ASP",
    "DTH": "THR",
    "DTR": "TRP",
    "DTY": "TYR",
    "DVA": "VAL",
    "EFC": "CYS",
    "FLA": "ALA",
    "FME": "MET",
    "GGL": "GLU",
    "GL3": "GLY",
    "GLZ": "GLY",
    "GMA": "GLU",
    "GSC": "GLY",
    "HAC": "ALA",
    "HAR": "ARG",
    "HIC": "HIS",
    "HIP": "HIS",
    "HMR": "ARG",
    "HPQ": "PHE",
    "HTR": "TRP",
    "HYP": "PRO",
    "IAS": "ASP",
    "IIL": "ILE",
    "IYR": "TYR",
    "KCX": "LYS",
    "LLP": "LYS",
    "LLY": "LYS",
    "LTR": "TRP",
    "LYM": "LYS",
    "LYZ": "LYS",
    "MAA": "ALA",
    "MEN": "ASN",
    "MHS": "HIS",
    "MIS": "SER",
    "MLE": "LEU",
    "MPQ": "GLY",
    "MSA": "GLY",
    "MSE": "MET",
    "MVA": "VAL",
    "NEM": "HIS",
    "NEP": "HIS",
    "NLE": "LEU",
    "NLN": "LEU",
    "NLP": "LEU",
    "NMC": "GLY",
    "OAS": "SER",
    "OCS": "CYS",
    "OMT": "MET",
    "PAQ": "TYR",
    "PCA": "GLU",
    "PEC": "CYS",
    "PHI": "PHE",
    "PHL": "PHE",
    "PR3": "CYS",
    "PRR": "ALA",
    "PTR": "TYR",
    "PYX": "CYS",
    "SAC": "SER",
    "SAR": "GLY",
    "SCH": "CYS",
    "SCS": "CYS",
    "SCY": "CYS",
    "SEL": "SER",
    "SEP": "SER",
    "SET": "SER",
    "SHC": "CYS",
    "SHR": "LYS",
    "SMC": "CYS",
    "SOC": "CYS",
    "STY": "TYR",
    "SVA": "SER",
    "TIH": "ALA",
    "TPL": "TRP",
    "TPO": "THR",
    "TPQ": "ALA",
    "TRG": "LYS",
    "TRO": "TRP",
    "TYB": "TYR",
    "TYI": "TYR",
    "TYQ": "TYR",
    "TYS": "TYR",
    "TYY": "TYR",
}


STD_AA_Domain = ProteinSequenceDomain(
    alphabet=STANDARD_AMINO_ACIDS_20,
    mapping={
        aa: torch.nn.functional.one_hot(idx, num_classes=20).float()
        for idx, aa in zip(torch.arange(20), STANDARD_AMINO_ACIDS_20)
    }
)

STD_AA_WITHUNK_Domain = ProteinSequenceDomain(
    alphabet=STANDARD_AMINO_ACIDS_20 + ['X'],
    mapping={
        aa: torch.nn.functional.one_hot(idx, num_classes=21).float()
        for idx, aa in zip(torch.arange(21), STANDARD_AMINO_ACIDS_20 + ['X'])
    }
)

STD_AA_WITHSPECIAL_Domain = ProteinSequenceDomain(
    alphabet=STANDARD_AMINO_ACIDS_22,
    mapping={
        aa: torch.nn.functional.one_hot(idx, num_classes=22).float()
        for idx, aa in zip(torch.arange(22), STANDARD_AMINO_ACIDS_22)
    }
)

STD_AA_WITHUNK_SPECIAL_Domain = ProteinSequenceDomain(
    alphabet=STANDARD_AMINO_ACIDS_22 + ['X'],
    mapping={
        aa: torch.nn.functional.one_hot(idx, num_classes=23).float()
        for idx, aa in zip(torch.arange(23), STANDARD_AMINO_ACIDS_22 + ['X'])
    }
)

STD_AA_SOTFLABEL_Domain = ProteinSequenceDomain(
    alphabet=STANDARD_AMINO_ACIDS_20 + ['X'],
    mapping=dict(
        {
            aa: torch.nn.functional.one_hot(idx, num_classes=20).float()
            for idx, aa in zip(torch.arange(20), STANDARD_AMINO_ACIDS_20)
        },
        **{'X': torch.full((20,), 1 / 20)}
    )
)
