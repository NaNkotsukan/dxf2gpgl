import numpy as np
from numpy.core.records import array
import itertools


class GPGL:
    def __init__(self, margin=10, blade_setting=100) -> None:
        self.max = np.tile(np.iinfo(np.int32).min, 2)
        self.min = np.tile(np.iinfo(np.int32).max, 2)
        self.path = []
        self.margin = margin
        self.blade_setting = blade_setting

    def push(self, array):
        self.max = np.maximum(array.max(axis=(0, 1)), self.max)
        self.min = np.minimum(array.min(axis=(0, 1)), self.min)
        self.path.append(array)

    def normalize(self):
        self.path = [x - self.min for x in self.path]
        self.max -= self.min
        self.min = np.zeros(2)

    def get_header(self):
        return "\x1b.v:TC1007,4,2,&100,100,100,^0,0,\0,0,TC1007,4,2,"

    def get_cmd(self):
        self.normalize()
        ret = []
        len(self.path)
        grid = itertools.product(range(self.margin+self.max[0]+self.blade_setting, self.margin+self.max[0]+self.blade_setting+(len(self.path)//(self.max[1]//(self.blade_setting*2))+1)*self.blade_setting*2, (self.blade_setting*2)), range(self.blade_setting, self.max[1], (self.blade_setting*2)))
        for x in self.path:
            setpoint = np.array(next(grid))
            self.max[0] // (self.blade_setting * 2)
            vec = x[0,1] - x[0,0]
            vec = vec * (self.blade_setting / 2 / np.linalg.norm(vec, ord=2))
            setpoint = np.vstack([setpoint-vec, setpoint+vec]).astype(np.int32)[None]
            ret.append(setpoint)
            ret.append(x)
        ret = np.vstack(ret).astype(np.int32)
        ret += self.margin

        out = self.get_header()
        for x in ret:
            out += f"M{x[0,0]},{x[0,1]},D{x[1,0]},{x[1,1]},"
        end = (self.max + self.margin * 2).astype(np.int32)
        out += f"M0,0,D0,{end[1]},D{end[0]},{end[1]},D{end[0]},0,D0,0,M0,0"
        return out
