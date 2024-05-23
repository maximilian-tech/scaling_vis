# acceleration_analysis.py

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sympy import symbols, Eq, solve, lambdify

class AccelerationAnalysis:
    def __init__(self, data):
        self.data = data
        self.nprocs_ = np.arange(1, 9, 1)
        self.frac_par_ = np.arange(0.01, 1.01, 0.01)
        self.nprocs, self.frac_par = np.meshgrid(self.nprocs_, self.frac_par_)
        self.accel_speedup = 1
        self.nodes = 1
        self.speedup = (((1 - self.frac_par) + self.nodes * self.frac_par) / 
                        ((1 - self.frac_par) + (self.frac_par / (self.accel_speedup * self.nprocs))))
        self.X, self.Y, self.Z = self.nprocs, self.frac_par, self.speedup
        self.frac_par_func = self._setup_symbolic_solver()

    def _setup_symbolic_solver(self):
        frac_par, speedup, nodes, accel_speedup, nprocs = symbols('frac_par speedup nodes accel_speedup nprocs')
        original_eq = Eq(speedup, ((1 - frac_par) + nodes * frac_par) / 
                         ((1 - frac_par) + (frac_par / (accel_speedup * nprocs))))
        cleared_eq = Eq(speedup * ((1 - frac_par) + (frac_par / (accel_speedup * nprocs))), 
                        (1 - frac_par) + nodes * frac_par)
        frac_par_solution = solve(cleared_eq, frac_par)
        frac_par_expr = frac_par_solution[0]
        return lambdify((speedup, nodes, accel_speedup, nprocs), frac_par_expr, 'numpy')

    def calculate_parallel_fraction(self):
        for gpu, runtimes in self.data["runtimes"].items():
            par_frac = []
            for idx, runtime in enumerate(runtimes):
                speedup_value = runtime / runtimes[0]
                nprocs_value = self.data["nproc"][idx]
                accel_speedup_value = 1
                nodes_value = 1
                if nprocs_value == 1: 
                    par_frac.append(1.0)
                    continue
                frac_par_calc = self.frac_par_func(speedup_value, nodes_value, accel_speedup_value, nprocs_value)
                par_frac.append(frac_par_calc)
            self.data["parallel_fractions"][gpu] = par_frac
            self.data["z_norm"][gpu] = [val / runtimes[0] for val in runtimes]

    def plot_surface(self):
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.plot_surface(self.X, self.Y, self.Z, edgecolor='royalblue', lw=0.5, rstride=10, cstride=1, alpha=0.2)
        ax.set(xlim=(1, 8), ylim=(0.01, 1), zlim=(1, 8), xlabel='GPUs', ylabel='Effective Parallel Fraction', zlabel='Speedup')

        for gpu, y_values in self.data['parallel_fractions'].items():
            z_values = self.data['z_norm'][gpu]
            ax.plot(self.data['nproc'], y_values, z_values, marker='o', label=gpu)
        
        plt.legend()
        plt.show()

def main():
    data = {
        "nproc": [1.0, 2.0, 4.0, 8.0],
        "parallel_fractions": {},  # To Be Filled
        "runtimes": {
            "A800": [2.0, 4.0, 8.2, 16.6],
            "RTX4090": [1.7, 3.2, 5.8, 11.5],
            "RTX3090_NV": [1.1, 2.4, 4.5, 7.5],
            "RTX3090": [1.1, 2.0, 4.1, 8.5],
            "fake": [1.0, 1.2, 1.25, 1.26],
        },
        "z_norm": {}
    }

    analysis = AccelerationAnalysis(data)
    analysis.calculate_parallel_fraction()
    analysis.plot_surface()
    print(data)

if __name__ == "__main__":
    main()
