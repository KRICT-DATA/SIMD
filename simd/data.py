import pandas
from itertools import chain
from tqdm import tqdm
from copy import deepcopy
from simd.chem import *


class MatData:
    def __init__(self, x, y, temp, form):
        self.x = x
        self.y = y
        self.temp = temp
        self.form = form
        self.sys_id = get_pristine_form(form)
        self.x_origin = deepcopy(x)


class Dataset:
    def __init__(self, data=None):
        self.data = list() if data is None else data

    def add(self, mat_data):
        self.data.append(mat_data)

    def x(self):
        return numpy.vstack([d.x for d in self.data])

    def y(self):
        return numpy.vstack([d.y for d in self.data])

    def temps(self):
        return numpy.vstack([d.temp for d in self.data])

    def sys_ids(self):
        return [d.sys_id for d in self.data]

    def forms(self):
        return [d.form for d in self.data]

    def merge(self, dataset_new):
        for d in dataset_new.data:
            self.data.append(d)

        return self

    def set_sys_info(self, sys_dict, k=1):
        elem_attrs = load_mendeleev_feats(['atomic_number', 'atomic_volume', 'atomic_weight'])
        sys_ids = sys_dict.sys_ids()
        sys_anchors = list()

        for sys_id in sys_ids:
            sys_anchors.append(get_form_vec(sys_id, elem_attrs))

        for i in range(0, len(self.data)):
            if self.data[i].sys_id in sys_ids:
                mat_sys = sys_dict.sys_dict[self.data[i].sys_id]
                self.data[i].x = numpy.hstack([self.data[i].x_origin, mat_sys.sys_vec, mat_sys.y_stat])
            else:
                anchor = get_form_vec(self.data[i].sys_id, elem_attrs)
                dists = numpy.sum((anchor - sys_anchors)**2, axis=1) + 1e-6

                idx_k = numpy.argsort(dists)[:k]
                simd_vec = list()
                weights_anc = list()

                for idx in idx_k:
                    mat_sys = sys_dict.sys_dict[sys_ids[idx]]
                    simd_vec.append(numpy.hstack([self.data[i].x_origin, mat_sys.sys_vec, mat_sys.y_stat]))
                    weights_anc.append(1 / (dists[idx] + 1e-6))
                weights_anc = (numpy.vstack(weights_anc) / sum(weights_anc)).reshape(-1, 1)

                self.data[i].x = numpy.sum(weights_anc * numpy.vstack(simd_vec), axis=0)

    def get_k_fold(self, k, random_seed=None):
        sub_data = split_list(self.data, k, random_seed=random_seed)
        k_folds = list()

        for i in range(0, k):
            data_train = concat_list(sub_data[:i], sub_data[i+1:])
            data_test = sub_data[i]
            k_folds.append([Dataset(data_train), Dataset(data_test)])

        return k_folds

    def get_k_fold_chem(self, k, random_seed=None):
        dict_sys = dict()
        for d in self.data:
            host_mat = get_pristine_form(d.form)

            if host_mat not in dict_sys.keys():
                dict_sys[host_mat] = list()
            dict_sys[host_mat].append(d)

        sub_data = split_list(list(dict_sys.values()), k, random_seed=random_seed)
        k_folds = list()

        for i in range(0, k):
            data_train = flat_list3d(sub_data[:i]) + flat_list3d(sub_data[i+1:])
            data_test = flat_list2d(sub_data[i])
            k_folds.append([Dataset(data_train), Dataset(data_test)])

        return k_folds

    def split(self, ratio):
        n_dataset1 = int(ratio * len(self.data))
        idx_rand = numpy.random.permutation(len(self.data))
        dataset1 = Dataset([self.data[idx] for idx in idx_rand[:n_dataset1]])
        dataset2 = Dataset([self.data[idx] for idx in idx_rand[n_dataset1:]])

        return dataset1, dataset2


def split_list(data_list, n_sub_lists, random_seed=None):
    if random_seed is not None:
        numpy.random.seed(random_seed)

    idx_rand = numpy.array_split(numpy.random.permutation(len(data_list)), n_sub_lists)
    sub_lists = list()

    for i in range(0, n_sub_lists):
        sub_lists.append([data_list[idx] for idx in idx_rand[i]])

    return sub_lists


def concat_list(l1, l2):
    return list(chain.from_iterable(l1 + l2))


def flat_list2d(l):
    if len(l) == 0:
        return l

    list_1d = list()

    for i in range(0, len(l)):
        for j in range(0, len(l[i])):
            list_1d.append(l[i][j])

    return list_1d


def flat_list3d(l):
    if len(l) == 0:
        return l

    list_1d = list()

    for i in range(0, len(l)):
        for j in range(0, len(l[i])):
            for k in range(0, len(l[i][j])):
                list_1d.append(l[i][j][k])

    return list_1d


def load_dataset(path_dataset, idx_form, idx_num_feats, idx_target, elem_attrs=None, log_target=False):
    data = numpy.array(pandas.read_excel(path_dataset))
    targets = data[:, idx_target].astype(float)
    temps = data[:, idx_num_feats]
    dataset = Dataset()

    if log_target:
        targets = numpy.log(targets + 1e-6)

    for i in tqdm(range(0, data.shape[0])):
        if get_pristine_form(data[i, idx_form]) == '':
            continue

        if elem_attrs is None:
            form_vec = numpy.hstack([get_form_vec_sparse(data[i, idx_form]), temps[i]])
        else:
            form_vec = numpy.hstack([get_form_vec(data[i, idx_form], elem_attrs), temps[i]])

        dataset.add(MatData(form_vec, targets[i], temps[i], data[i, idx_form]))

    return dataset
