from ase.data import atomic_numbers, covalent_radii
from ase.data.vdw_alvarez import vdw_radii

els = ['Co', 'C', 'C', 'H','H', 'Ti', 'Mn', 'Pu', 'U', 'Th' , "Np", "O"]
eqs = [1.98, 2.09, 2.41, 2.28, 2.44, 2.16, 1.95, 3.149, 3.23, 3.41, 3.18, 3.68]
for i in range(len(els)):
    el = els[i]
    alat_eql = eqs[i]
    an = atomic_numbers[el]
    alat_cov = 1.414 * covalent_radii[an] 
    alat_vdw = 1.414 * vdw_radii[an] 
    print(el)
    print(f"\tALAT_COV {alat_cov}")
    print(f"\tALAT_VDW {alat_vdw}")
    print(f"\tALAT_EQL {alat_eql}")
    print(f"\tERR_COV% {100 * (1 - alat_cov/alat_eql)} {'*' if alat_cov < alat_eql else ''}")
    print(f"\tERR_VDW% {100 * (1 - alat_vdw/alat_eql)} {'*' if alat_vdw < alat_eql else ''}")
    print("----------------")
