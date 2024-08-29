# Data Structure for protein biology

## Sequence Data

### Data Definitions

Sequence data contains protein sequence, DNA sequence with also RNA sequence. All the
sequence type is such a datatype with a predifined:

1. Partial ordering relation. This ensures the sequence is indeed a sequence.
2. Encode alphabet. This ensures the sub-sequence level entity has its mapping to structure 
data. For example, a X stands for an amino acid with atom type just like Ala but a loss free 
virtual atom after CB atom.
3. Decode alphabet. This ensures the sub-sequence level entity meet the generation objects with 
an inner relation to make the final self-supervised style training a multi-label style training.

It is very important that the encode alphabet may not be exactly the same with decode alphabet.