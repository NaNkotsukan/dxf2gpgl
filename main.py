from numpy import mod
from figure import Figure, Figure2
import solver
from gpgl import GPGL
import numpy as np
from show import save

def main():
    fig = Figure("test_pattern.dxf")
    point = None
    route = None
    gpgl = GPGL(1000, 100)
    for i in range(len(fig)):
        point, dist = fig.get_path(i)
        model = solver.Salesman(dist)
        energy, route = model.solve()
        gpgl.push(point[route,:,::-1].astype(np.int32))
        save(f"pattern{i}", point[route].astype(np.int32), fig.size)
    cmd = gpgl.get_cmd()
    with open("test_pattern.plt", "w") as f:
        f.write(cmd)
    print(cmd)
    

if __name__ == '__main__':
    main()
