from sklearn.preprocessing import scale
from simd.chem import *


class MatSystem:
    def __init__(self, sys_id):
        self.sys_id = sys_id
        self.x = list()
        self.y = list()
        self.y_stat = None
        self.sys_vec = None

    def add(self, x, y):
        self.x.append(x)
        self.y.append(y)

    def get_sys_vec(self):
        try:
            return numpy.linalg.lstsq(numpy.vstack(self.x), numpy.vstack(self.y), rcond=-1)[0].reshape(1, -1)
        except:
            return numpy.zeros(self.x[0].shape[0]).reshape(1, -1)

    def get_y_stat(self, temp):
        if temp < 300:
            return self.y_stat[:4]
        elif 300 <= temp < 600:
            return self.y_stat[4:8]
        elif 600 <= temp < 800:
            return self.y_stat[8:12]
        elif 800 <= temp:
            return self.y_stat[12:]

    def set_y_stat(self, y, temps):
        y_vals = [[], [], [], []]
        y_stat = list()

        for i in range(0, len(y)):
            if temps[i] < 300:
                y_vals[0].append(y[i])
            elif 300 <= temps[i] < 600:
                y_vals[1].append(y[i])
            elif 600 <= temps[i] < 800:
                y_vals[2].append(y[i])
            elif 800 <= temps[i]:
                y_vals[3].append(y[i])

        for i in range(0, len(y_vals)):
            if len(y_vals[i]) > 0:
                y_stat.append(numpy.mean(y_vals[i]))
                y_stat.append(numpy.std(y_vals[i]))
                y_stat.append(numpy.min(y_vals[i]))
                y_stat.append(numpy.max(y_vals[i]))
            else:
                y_stat.append(0)
                y_stat.append(0)
                y_stat.append(0)
                y_stat.append(0)

        self.y_stat = numpy.array(y_stat)


class MatSysDict:
    def __init__(self, src_dataset):
        self.anc_attrs = scale(load_mendeleev_feats(['atomic_number', 'atomic_volume', 'atomic_weight']))
        self.sys_dict = dict()
        self.sys_ancs = list()

        self.__init_sys_dict(src_dataset)
        self.__set_sys_vecs()
        self.__set_sys_y_stat(src_dataset)
        self.__set_sys_ancs()

    def __init_sys_dict(self, src_dataset):
        for d in src_dataset.data:
            sys_id = d.sys_id

            if sys_id == '':
                continue

            if sys_id not in self.sys_dict.keys():
                self.sys_dict[sys_id] = MatSystem(sys_id)

            self.sys_dict[sys_id].add(d.x_origin, d.y)

    def __set_sys_vecs(self):
        sys_ids = self.sys_ids()
        sys_vecs = scale(numpy.vstack([self.sys_dict[sys_id].get_sys_vec() for sys_id in sys_ids]))

        for i in range(0, len(sys_ids)):
            mat_sys = self.sys_dict[sys_ids[i]]
            mat_sys.sys_vec = sys_vecs[i]

    def __set_sys_y_stat(self, src_dataset):
        sys_data = dict()

        for d in src_dataset.data:
            if d.sys_id not in sys_data.keys():
                sys_data[d.sys_id] = [[], []]
            sys_data[d.sys_id][0].append(d.y)
            sys_data[d.sys_id][1].append(d.temp)

        for sys_id in self.sys_dict.keys():
            self.sys_dict[sys_id].set_y_stat(sys_data[sys_id][0], sys_data[sys_id][1])

    def __set_sys_ancs(self):
        self.sys_ancs = [get_form_vec(sys_id, self.anc_attrs) for sys_id in self.sys_ids()]

    def sys_ids(self):
        return list(self.sys_dict.keys())

    def sys_vec(self, sys_id):
        return self.sys_dict[sys_id].sys_vec

    def set_sys_avg_y(self, sys_ids, y, temps):
        for sys_id in self.sys_dict.keys():
            self.sys_dict[sys_id].set_y_stat(sys_ids, y, temps)

    def get_sys_x(self, form_vec, temp):
        sys_ids = self.sys_ids()

        form = ''
        for i in range(0, form_vec.shape[0]):
            if form_vec[i] > 0:
                form += atom_syms[i + 1] + str(form_vec[i])
        sys_id = get_pristine_form(form)

        if sys_id == '':
            mat_sys = self.sys_dict[sys_ids[0]]
            return numpy.zeros(form_vec.shape[0] + mat_sys.sys_vec.shape[0] + mat_sys.y_stat.shape[0] + 1)

        if sys_id in sys_ids:
            mat_sys = self.sys_dict[sys_id]
        else:
            anchor = get_form_vec(sys_id, self.anc_attrs)
            dists = numpy.sum((anchor - self.sys_ancs)**2, axis=1)
            mat_sys = self.sys_dict[sys_ids[numpy.argmin(dists)]]

        return numpy.hstack([form_vec, temp, mat_sys.sys_vec, mat_sys.y_stat])
