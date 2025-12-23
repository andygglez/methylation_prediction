import numpy as np



chr19 = np.load("chr19.npz", allow_pickle=True)

value_list = list(zip(range(len(chr19['methyl'])), chr19['methyl']))

value_list_filtered = list(filter(lambda x: x[1]>0.8 or x[1]<0.2, value_list))
indexes = list(map(lambda x: x[0], value_list_filtered))

dna = chr19['dna'][indexes]
histone = chr19['histone'][indexes]
methyl = chr19['methyl'][indexes]
coords = chr19['coords'][indexes]
histone_names = chr19['histone_names'][indexes]


np.savez_compressed(
    "chr19.filtered.npz",
    dna=dna,
    histone=histone,
    methyl=methyl,
    coords=coords,
    histone_names=histone_names
)