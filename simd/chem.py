import numpy
import json
from mendeleev.fetch import fetch_table
from chemparse import parse_formula


atom_nums = {'H': 1, 'He': 2, 'Li': 3, 'Be': 4, 'B': 5, 'C': 6, 'N': 7, 'O': 8, 'F': 9, 'Ne': 10,
             'Na': 11, 'Mg': 12, 'Al': 13, 'Si': 14, 'P': 15, 'S': 16, 'Cl': 17, 'Ar': 18, 'K': 19, 'Ca': 20,
             'Sc': 21, 'Ti': 22, 'V': 23, 'Cr': 24, 'Mn': 25, 'Fe': 26, 'Co': 27, 'Ni': 28, 'Cu': 29, 'Zn': 30,
             'Ga': 31, 'Ge': 32, 'As': 33, 'Se': 34, 'Br': 35, 'Kr': 36, 'Rb': 37, 'Sr': 38, 'Y': 39, 'Zr': 40,
             'Nb': 41, 'Mo': 42, 'Tc': 43, 'Ru': 44, 'Rh': 45, 'Pd': 46, 'Ag': 47, 'Cd': 48, 'In': 49, 'Sn': 50,
             'Sb': 51, 'Te': 52, 'I': 53, 'Xe': 54, 'Cs': 55, 'Ba': 56, 'La': 57, 'Ce': 58, 'Pr': 59, 'Nd': 60,
             'Pm': 61, 'Sm': 62, 'Eu': 63, 'Gd': 64, 'Tb': 65, 'Dy': 66, 'Ho': 67, 'Er': 68, 'Tm': 69, 'Yb': 70,
             'Lu': 71, 'Hf': 72, 'Ta': 73, 'W': 74, 'Re': 75, 'Os': 76, 'Ir': 77, 'Pt': 78, 'Au': 79, 'Hg': 80,
             'Tl': 81, 'Pb': 82, 'Bi': 83, 'Po': 84, 'At': 85, 'Rn': 86, 'Fr': 87, 'Ra': 88, 'Ac': 89, 'Th': 90,
             'Pa': 91, 'U': 92, 'Np': 93, 'Pu': 94, 'Am': 95, 'Cm': 96, 'Bk': 97, 'Cf': 98, 'Es': 99, 'Fm': 100}
atom_syms = {v: k for k, v in atom_nums.items()}
first_ion_energies = [1312, 2372.3, 520.2, 899.5, 800.6, 1086.5, 1402.3, 1313.9, 1681, 2080.7,
                      495.8, 737.7, 577.5, 786.5, 1011.8, 999.6, 1251.2, 1520.6, 418.8, 589.8,
                      633.1, 658.8, 650.9, 652.9, 717.3, 762.5, 760.4, 737.1, 745.5, 906.4,
                      578.8, 762, 947, 941, 1139.9, 1350.8, 403, 549.5, 600, 640.1,
                      652.1, 684.3, 702, 710.2, 719.7, 804.4, 731, 867.8, 558.3, 708.6,
                      834, 869.3, 1008.4, 1170.4, 375.7, 502.9, 538.1, 534.4, 527, 533.1,
                      540, 544.5, 547.1, 593.4, 565.8, 573, 581, 589.3, 596.7, 603.4,
                      523.5, 658.5, 761, 770, 760, 840, 880, 870, 890.1, 1007.1,
                      589.4, 715.6, 703, 812.1, 899.003, 1037, 380, 509.3, 499, 587,
                      568, 597.6, 604.5, 584.7, 578, 581, 601, 608, 619, 627,
                      635, 642, 470, 580, 665, 757, 740, 730, 800, 960,
                      1020, 1155, 707.2, 832.2, 538.3, 663.9, 736.9, 860.1, 463.1, 563.3]


def load_mendeleev_feats(elem_feat_names):
    return numpy.nan_to_num(numpy.array(fetch_table('elements')[elem_feat_names]))


def load_elem_embeddings(path_elem_attr):
    elem_attrs = list()
    with open(path_elem_attr) as json_file:
        elem_attr = json.load(json_file)
        for elem in atom_nums.keys():
            elem_attrs.append(numpy.array(elem_attr[elem]))

    return numpy.vstack(elem_attrs)


def get_form_vec(form, elem_attrs):
    wt_sum_feats = numpy.zeros(elem_attrs.shape[1])
    list_atom_feats = list()

    form_dict = parse_formula(form)
    sum_elem_nums = numpy.sum([float(form_dict[e]) for e in form_dict.keys()])

    for e in form_dict.keys():
        atom_feats = elem_attrs[atom_nums[e] - 1, :]
        list_atom_feats.append(atom_feats)
        wt_sum_feats += (float(form_dict[e]) / sum_elem_nums) * atom_feats

    form_vec = numpy.hstack([wt_sum_feats,
                             numpy.std(list_atom_feats, axis=0),
                             numpy.max(list_atom_feats, axis=0),
                             numpy.min(list_atom_feats, axis=0)])

    return form_vec


def get_form_vec_sparse(form):
    form_vec = numpy.zeros(len(atom_nums.keys()))
    form_dict = parse_formula(form)

    for e in form_dict.keys():
        form_vec[atom_nums[e] - 1] = form_dict[e]

    return form_vec


def get_host_form(form):
    form_dict = parse_formula(form)
    sorted_elems = sorted(form_dict.keys(), key=lambda x: x.lower())
    form_host_mat = ''

    for e in sorted_elems:
        if form_dict[e] >= 0.5:
            form_host_mat += e + str(round(form_dict[e]))

    return form_host_mat


def get_pristine_form(form):
    form_dict = parse_formula(form)
    pristine_mat = dict()

    sorted_keys = sorted(form_dict.keys(), key=lambda x: x.lower())
    for e in sorted_keys:
        if form_dict[e] >= 0.5:
            pristine_mat[e] = round(form_dict[e])

    form_pristine_mat = ''
    for e in pristine_mat:
        if pristine_mat[e] > 0:
            form_pristine_mat += e + str(pristine_mat[e])

    return form_pristine_mat


def encoding_to_form(encoding, elems):
    form = ''

    for i in range(0, encoding.shape[0]):
        if encoding[i] > 0:
            form += elems[i] + str(encoding[i])

    return form


def parse_encoding(encoding, elems):
    output = numpy.zeros(len(atom_nums.keys()))

    for i in range(0, encoding.shape[0]):
        output[atom_nums[elems[i]] - 1] = encoding[i]

    return output
