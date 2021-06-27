import ezdxf
import numpy as np
from scipy.sparse.csgraph import connected_components
from scipy.sparse import csr_matrix, csgraph


class Figure:
    def __init__(self, path, ratio=1000, margin=1) -> None:
        doc = ezdxf.readfile(path)
        msp = doc.modelspace()
        lines = msp.query('LINE')
        lines = np.array([(line.dxf.start, line.dxf.end) for line in lines]) * ratio
        vector = lines[:,0] - lines[:,1]
        valid_axis = [not np.allclose(vector[:,i], 0) for i in range(3)]
        lines = lines[:,:,valid_axis]
        vector = vector[:,valid_axis]
        for i in np.argwhere(np.arctan2(vector[:,0], vector[:,1]) <= 0):
            lines[i] = lines[i,::-1]
        vector = lines[:,0] - lines[:,1]
        norm = np.linalg.norm(vector, ord=2, axis=-1, keepdims=True)
        normalized_vec = vector/norm
        adj_matrix = np.dot(normalized_vec, normalized_vec.T)
        self.n_components, labels = connected_components(np.isclose(np.abs(adj_matrix),1))
        direction = [list() for _ in range(self.n_components)]
        for i, l in enumerate(labels):
            direction[l].append(i)
        lines = lines - lines.min(axis=(0,1), keepdims=True)
        self.lines = lines
        self.vector = vector
        self.direction = direction
        self.size = lines.max(axis=(0,1)).astype(np.int32)
        self.margin = margin
    
    def get_path(self, i):
        lines = self.lines[self.direction[i]]
        vector = self.vector[self.direction[i]]
        start = lines[:,0]
        end = start - (np.trunc(vector) + np.sign(vector) * self.margin)
        path = np.expand_dims(end, axis=1) - np.expand_dims(start, axis=0)
        distance = np.linalg.norm(path, ord=2, axis=-1)
        return np.stack((start, end), axis=1), distance

    def __len__(self):
        return self.n_components


class Figure2(Figure):
    def __init__(self, path, ratio=1000, margin=5) -> None:
        super().__init__(path, ratio=ratio, margin=margin)

    def get_path(self, i):
        lines = self.lines[self.direction[i//2]]
        vector = self.vector[self.direction[i//2]]
        if i % 2 == 0:
            start = lines[:,0]
            end = start - (np.trunc(vector * 0.5) + np.sign(vector) * self.margin)
        else:
            start = lines[:,1]
            end = start + (np.trunc(vector * 0.5) + np.sign(vector) * self.margin)
        path = np.expand_dims(end, axis=1) - np.expand_dims(start, axis=0)
        distance = np.linalg.norm(path, ord=2, axis=-1)
        return np.stack((start, end), axis=1), distance

    def __len__(self):
        return self.n_components * 2

