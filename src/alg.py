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


    def subtask_parts(self, similarities: np.ndarray) -> np.ndarray:
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
        similarities = self.parts_similarities()
        clusters_parts = self.subtask_parts(similarities)
        clusters_machines = self.subtask_machines(clusters_parts)
        return clusters_parts, clusters_machines


def read_data(path: str) -> Tuple[int, int, List]:
    with open(path, 'r') as f:
        m_len, p_len = map(int, f.readline().split())
        lines = [line for line in f]

    mp_data = [list(map(int, lines[i].split()))[1:] for i in range(m_len)]
    return m_len, p_len, mp_data


if __name__ == '__main__':
    m1, p1, mtx1 = read_data('benchmarks/24x40.txt')
    alg = SimulatedAnnealingAlgorithm(m1, p1, mtx1)
    print(alg.initial_solution())
