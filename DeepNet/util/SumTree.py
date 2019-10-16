import numpy

class SumTree:
    write = 0

    def __init__(self, capacity, load, load_path):
        if load:
            self.capacity, self.tree, self.data = self.load_sumTree(load_path)

        else:
            self.capacity = capacity
            self.tree = numpy.zeros( 2*capacity - 1 )
            self.data = numpy.zeros( capacity, dtype=object )

    def _propagate(self, idx, change):
        parent = (idx - 1) // 2

        self.tree[parent] += change

        if parent != 0:
            self._propagate(parent, change)

    def _retrieve(self, idx, s):
        left = 2 * idx + 1
        right = left + 1

        if left >= len(self.tree):
            return idx

        if s <= self.tree[left]:
            return self._retrieve(left, s)
        else:
            return self._retrieve(right, s-self.tree[left])

    def total(self):
        return self.tree[0]

    def add(self, p, data):
        idx = self.write + self.capacity - 1

        self.data[self.write] = data
        self.update(idx, p)

        self.write += 1
        if self.write >= self.capacity:
            self.write = 0

    def update(self, idx, p):
        change = p - self.tree[idx]

        self.tree[idx] = p
        self._propagate(idx, change)

    def get(self, s):
        idx = self._retrieve(0, s)
        dataIdx = idx - self.capacity + 1

        return (idx, self.tree[idx], self.data[dataIdx])

    def load_sumTree(self, path):
        sumTree = numpy.load(path+'sumTree.npy')
        ccc = numpy.shape(sumTree)
        capacity = sumTree.item().get('capacity')
        data = sumTree.item().get('data')
        tree = sumTree.item().get('tree')

        return capacity, tree, data


    def save_sumTree(self, path):
        sumTree={}
        sumTree['capacity'] = self.capacity
        sumTree['data'] = self.data
        sumTree['tree'] = self.tree

        numpy.save(path+'sumTree.npy', sumTree)

