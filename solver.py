from numpy import cos
from numpy.linalg.linalg import solve
from amplify import (
    BinaryPoly,
    BinaryQuadraticModel,
    sum_poly,
    gen_symbols,
    Solver,
    decode_solution,
)
from amplify.client import FixstarsClient
from amplify.constraint import equal_to
import numpy as np
from concurrent.futures import ThreadPoolExecutor
import config

client = FixstarsClient()
client.token = config.TOKEN
client.parameters.timeout = 10000
solver = Solver(client)

class Salesman:
    def __init__(self, distances, ncity=None):
        if ncity is None:
            ncity = len(distances)
        q = gen_symbols(BinaryPoly, ncity, ncity)
        selector = np.arange(ncity)
        d_min = np.array([d[selector!=i].min() for i, d in enumerate(distances)])
        distances -= d_min
        cost = sum_poly(
            ncity,
            lambda n: sum_poly(
                ncity,
                lambda i: sum_poly(
                    ncity,
                    lambda j: (distances[i][j] if i != j else 0) * q[n][i] * q[(n + 1) % ncity][j]
                ),
            ),
        )
        row_constraints = [
            equal_to(sum_poly([q[n][i] for i in range(ncity)]), 1) for n in range(ncity)
        ]
        col_constraints = [
            equal_to(sum_poly([q[n][i] for n in range(ncity)]), 1) for i in range(ncity)
        ]
        constraints = sum(row_constraints) + sum(col_constraints)
        self.model = lambda k:cost + constraints * (distances.max() * k)
        self.q = q

    def solve(self):
        for i in range(10):
            try: 
                result = solver.solve(self.model(i*0.1))
                if len(result) == 0:
                    raise RuntimeError("Any one of constraints is not satisfied.")
                break
            except:
                continue
        energy, values = result[0].energy, result[0].values
        q_values = decode_solution(self.q, values, 1)
        route = np.where(np.array(q_values) == 1)[1]
        return energy, route


