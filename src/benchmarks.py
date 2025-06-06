import os
import time
import pandas as pd
from src.alg import *


class Benchmark:
    def __init__(self, algorithm, bench_dir: str, runs: int = 3):
        self.algorithm = algorithm
        self.runs = runs
        self.bench_dir = bench_dir
        self.results = []

    def run_all(self):
        files = [f for f in os.listdir(self.bench_dir)]
        for file in sorted(files):
            print(f'Benchmarking {file} ...')
            full_path = os.path.join(self.bench_dir, file)
            self.run_one(full_path, file)

        df = pd.DataFrame(self.results).sort_values(by=['benchmark_id', 'alg'], ascending=True)
        df.to_csv('results.csv', index=False)

    def run_one(self, path: str, benchmark: str):
        m, p, mtx = read_data(path)

        best_total_f = float('-inf')
        best_parts, best_machines = [], []
        total_time = 0
        for _ in range(self.runs):
            alg = self.algorithm(m, p, mtx)
            start = time.time()

            f_opt, c, parts, machines = alg.solve()

            total_time += time.time() - start
            if f_opt > best_total_f:
                best_total_f = f_opt
                best_parts, best_machines = parts.copy(), machines.copy()

        avg_time = total_time / self.runs
        self.results.append({
            'benchmark_id': int(benchmark[:-7]),
            'alg': self.algorithm.__name__,
            'benchmark': benchmark,
            'best_total_cost': round(best_total_f, 7),
            'avg_time_sec': round(avg_time, 7),
            'best_parts': " ".join(map(str, best_parts)),
            'best_machines': " ".join(map(str, best_machines))
        })
        self.save_solution("results/"+benchmark[:-4] + ".sol", best_machines, best_parts)

    def save_solution(self, file_path: str, best_machines, best_parts) -> None:
        with open(file_path, 'w') as file:
            file.write(" ".join(map(str, best_machines)))
            file.write("\n")
            file.write(" ".join(map(str, best_parts)))


if __name__ == '__main__':
    benchmark = Benchmark(SimulatedAnnealingAlgorithm, 'benchmarks', 5)
    benchmark.run_all()