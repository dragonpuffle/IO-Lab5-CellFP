import os

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__)))
bench_path = os.path.join(project_root, 'plots')

data = pd.read_csv('src/results.csv')

sns.set_style("whitegrid")
plt.figure(figsize=(12, 8))

plt.figure(figsize=(14, 8))
sns.barplot(x='benchmark', y='avg_time_sec', hue='alg', data=data)
plt.yscale('log')
plt.title('Execution Time by Benchmark and Algorithm (seconds)')
plt.xlabel('Benchmark ID')
plt.ylabel('Time (seconds)')
plt.legend(title='Algorithm', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.savefig(os.path.join(bench_path, 'execution_time.png'), dpi=300, bbox_inches='tight')
plt.close()

plt.figure(figsize=(14, 8))
sns.barplot(x='benchmark', y='best_total_cost', hue='alg', data=data)
plt.yscale('log')
plt.title('Algorithm Cost by Benchmark and Algorithm')
plt.xlabel('Benchmark ID')
plt.ylabel('Cost')
plt.legend(title='Algorithm', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.savefig(os.path.join(bench_path, 'algorithm_cost.png'), dpi=300, bbox_inches='tight')
plt.close()
