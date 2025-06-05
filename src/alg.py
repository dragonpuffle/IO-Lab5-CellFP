from typing import Tuple, List

import numpy as np


class SimulatedAnnealingAlgorithm:
    def __init__(self, m: int, p: int, mp_data: list, num_clusters: int = 2):
        self.m = m
        self.p = p
        self.mp_data = self.preprocess_data(mp_data)
        self.num_clusters = num_clusters


    def preprocess_data(self, mp_data: list) -> np.ndarray:
        # нужно будет перевести из начального вида в матрицу m*p
        new_mp = np.zeros((self.m, self.p))
        for machine in range(self.m):
            for part in mp_data[machine]:
                new_mp[machine, part-1] = 1

        return new_mp


    def objective_function(self, n1: int, n1_out: int, n0_in: int) -> float:
        return (n1 - n1_out) / (n1 + n0_in)


    def calc_f(self, parts: np.ndarray, machines: np.ndarray) -> float:
        n1 = np.sum(self.mp_data)
        n1_out, n0_in = 0, 0

        for i in range(self.m):
            for j in range(self.p):
                if self.mp_data[i][j] == 1 and machines[i] != parts[j]:
                    n1_out += 1 # possibly wrong
                elif self.mp_data[i][j] == 0 and machines[i] == parts[j]:
                    n0_in += 1 # possibly wrong

        return self.objective_function(n1, n1_out, n0_in)


    def parts_similarities(self) -> np.ndarray:
        similarities = np.zeros((self.p, self.p))

        for i in range(self.p):
            for j in range(self.p):
                if i == j:
                    similarities[i, j] = 1.0
                    continue

                mp_col_i = self.mp_data[:, i]
                mp_col_j = self.mp_data[:, j]

                a_ij = np.sum(np.logical_and(mp_col_i == 1, mp_col_j == 1))
                b_ij = np.sum(np.logical_and(mp_col_i == 1, mp_col_j == 0))
                c_ij = np.sum(np.logical_and(mp_col_i == 0, mp_col_j == 1))

                abc = a_ij + b_ij + c_ij
                similarities[i, j] = a_ij / abc if abc != 0 else 0.0

        return similarities


    def subtask_parts(self) -> np.ndarray:
        similarities = self.parts_similarities()

        assigned = set()
        clusters = []

        for cluster_id in range(self.num_clusters):
            unassigned = [i for i in range(self.p) if i not in assigned]
            if not unassigned:
                break

            current_part = unassigned[0]
            current_cluster = [current_part]
            assigned.add(current_part)

            for _ in range(self.p // self.num_clusters):
                candidates = [candidate for candidate in range(self.p) if candidate not in assigned]
                if not candidates:
                    break
                best_candidate = max(candidates, key=lambda candidate: np.mean([similarities[candidate, k] for k in current_cluster]))
                current_cluster.append(best_candidate)
                assigned.add(best_candidate)

            clusters.append(current_cluster)

        leftovers = [n_candidate for n_candidate in range(self.p) if n_candidate not in assigned]

        is_leftovers_new_cluster = True
        if leftovers:
            if is_leftovers_new_cluster:
                clusters.append(leftovers)
                self.num_clusters += 1
            else:
                for i, part in enumerate(leftovers):
                    clusters[i % self.num_clusters].append(part)

        clusters_parts = np.full(self.p, -1)

        for cluster_id, part_ids in enumerate(clusters):
            for part_id in part_ids:
                clusters_parts[part_id] = cluster_id

        return clusters_parts


    def subtask_machines(self, clusters_parts: np.ndarray) -> np.ndarray:
        clusters_machines = np.full(self.m, -1)

        for machine in range(self.m):
            best_cluster = -1
            best_cost = np.inf

            mp_row = self.mp_data[machine]
            for cluster_id in range(self.num_clusters):
                part_ids = np.where(clusters_parts == cluster_id)[0]

                part_mask = np.zeros(self.p, dtype=bool)
                part_mask[part_ids] = True

                v = np.sum(mp_row[~part_mask])
                e = np.sum((mp_row == 0) & part_mask)

                cost = v + e
                if cost < best_cost:
                    best_cost = cost
                    best_cluster = cluster_id

            clusters_machines[machine] = best_cluster

        return clusters_machines


    def initial_solution(self):
        clusters_parts = self.subtask_parts()
        clusters_machines = self.subtask_machines(clusters_parts)
        return clusters_parts, clusters_machines


    def single_move(self, clusters_parts: np.ndarray, clusters_machines: np.ndarray, mode: str='part')\
            -> Tuple[np.ndarray, np.ndarray]:
        if mode == 'part':
            clusters = clusters_parts.copy()
        else:
            clusters = clusters_machines.copy()

        item_idx = np.random.choice(len(clusters))
        current_cluster = clusters[item_idx]

        possible_clusters = list(set(clusters))
        if len(possible_clusters) < 2:
            return clusters_parts, clusters_machines

        target_cluster = np.random.choice([cluster for cluster in possible_clusters if cluster != current_cluster])
        clusters[item_idx] = target_cluster

        if mode == 'part':
            return clusters, clusters_machines
        else:
            return clusters_parts, clusters


    def exchange_move(self, clusters_parts: np.ndarray, clusters_machines: np.ndarray, mode: str='part')\
            -> Tuple[np.ndarray, np.ndarray]:
        if mode == 'part':
            clusters = clusters_parts.copy()
        else:
            clusters = clusters_machines.copy()

        indices = np.arange(len(clusters))
        np.random.shuffle(indices)

        for i in indices:
            for j in indices:
                if i != j and clusters[i] != clusters[j]:
                    clusters[i], clusters[j] = clusters[j], clusters[i]
                    if mode == 'part':
                        return clusters, clusters_machines
                    else:
                        return clusters_parts, clusters

        return clusters_parts, clusters_machines


    def solve(self, t0: float = 0.85, tf: float = 0.1, alpha: float = 0.9, l: int = 100, d: int = 10)\
            -> Tuple[float, float, np.ndarray, np.ndarray]:
        # 1
        s_parts_curr, s_machines_curr = self.initial_solution()
        s_parts_best_current_cell, s_machines_best_current_cell = s_parts_curr.copy(), s_machines_curr.copy()
        s_parts_best_total, s_machines_best_total = s_parts_curr.copy(), s_machines_curr.copy()

        f_curr = self.calc_f(s_parts_curr, s_machines_curr)
        f_best_current_cell = 0
        f_best_total = 0

        c_best = self.num_clusters
        c_max = min(self.m, self.p)

        # 2
        counter = 0
        counter_mc, counter_trapped, counter_stagnant = 0, 0, 0
        t = t0

        while True:
            # 3
            while counter_mc < l and counter_trapped < l/2:
                # 3.1
                s_parts_new, s_machines_new = self.single_move(s_parts_curr, s_machines_curr, mode='part')

                # 3.2
                if counter % d ==0:
                    s_parts_new, s_machines_new = self.exchange_move(s_parts_new, s_machines_new, mode='part')

                # 3.3
                s_parts_neighbor = s_parts_new.copy()
                s_machines_neighbor = self.subtask_machines(s_parts_neighbor)
                f_neighbor = self.calc_f(s_parts_neighbor, s_machines_neighbor)

                # 3.4
                if f_neighbor > f_best_current_cell:
                    s_parts_best_current_cell, s_machines_best_current_cell = (s_parts_neighbor.copy(),
                                                                               s_machines_neighbor.copy())
                    f_best_current_cell = f_neighbor

                    s_parts_curr, s_machines_curr = s_parts_neighbor.copy(),s_machines_neighbor.copy()
                    f_curr = f_neighbor

                    counter_stagnant = 0
                    counter_mc += 1
                    continue

                # 3.5
                if f_neighbor == f_best_current_cell:
                    s_parts_curr, s_machines_curr = s_parts_neighbor.copy(),s_machines_neighbor.copy()
                    f_curr = f_neighbor

                    counter_stagnant += 1
                    counter_mc += 1
                    continue

                # 3.6
                delta = f_neighbor - f_curr
                prob = np.exp(-delta / t)
                if prob > np.random.rand():
                    s_parts_curr, s_machines_curr = s_parts_neighbor.copy(),s_machines_neighbor.copy()
                    f_curr = f_neighbor

                    counter_trapped = 0
                else:
                    counter_trapped += 1

                # 3.7
                counter_mc += 1
                continue

            # 4
            if t <= tf or counter_stagnant > l:
                # 5 have to be f_best_curr > f_best_total, but with >= may happen better solution through some times
                if f_best_current_cell >= f_best_total and self.num_clusters < c_max:
                    print('cells =', self.num_clusters, 'f_value =', f_best_current_cell)
                    s_parts_best_total, s_machines_best_total = (s_parts_best_current_cell.copy(),
                                                                 s_machines_best_current_cell.copy())
                    f_best_total = f_best_current_cell

                    c_best = self.num_clusters
                    self.num_clusters += 1

                    # go to 2
                    s_parts_curr, s_machines_curr = self.initial_solution()
                    counter = 0
                    counter_mc, counter_trapped, counter_stagnant = 0, 0, 0
                    t = t0
                else:
                    return f_best_total, c_best, s_parts_best_total, s_machines_best_total
            else:
                t *= alpha
                counter_mc = 0
                counter += 1


def read_data(path: str) -> Tuple[int, int, List]:
    with open(path, 'r') as f:
        m_len, p_len = map(int, f.readline().split())
        lines = [line for line in f]

    mp_data = [list(map(int, lines[i].split()))[1:] for i in range(m_len)]
    return m_len, p_len, mp_data


if __name__ == '__main__':
    m1, p1, mtx1 = read_data('benchmarks/24x40.txt')
    alg = SimulatedAnnealingAlgorithm(m1, p1, mtx1)
    print(alg.solve())
