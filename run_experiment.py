import numpy as np
import time
import argparse
import pandas as pd # ★pandasをインポート
import os # ★osをインポート

# 作成したパッケージからクラスをインポート
from problems import SphereFunction, RosenbrockFunction, AckleyFunction
from algorithms import PSO,EFWA, DynFWA, HCFWA

# 実行可能なアルゴリズムを管理する辞書
ALGORITHM_MAP = {
    'pso': PSO,
    'efwa': EFWA,
    'dynfwa': DynFWA,
    'hcfwa': HCFWA,
}

# ★★★ 追加：実行可能な問題を管理する辞書 ★★★
PROBLEM_MAP = {
    'sphere': SphereFunction,
    'rosenbrock': RosenbrockFunction,
    'ackley': AckleyFunction,
}

def main():
    parser = argparse.ArgumentParser(description="Run an optimization experiment.")
    parser.add_argument('--algorithm', type=str, required=True, choices=ALGORITHM_MAP.keys(), help='The algorithm to run.')
    parser.add_argument('--problem', type=str, required=True, choices=PROBLEM_MAP.keys(), help='The problem function to optimize.')
    parser.add_argument('--dim', type=int, default=20, help='Dimension of the problem.')
    parser.add_argument('--runs', type=int, default=5, help='Number of runs.')
    
    args = parser.parse_args()
    
    ProblemClass = PROBLEM_MAP[args.problem]
    problem = ProblemClass(dimension=args.dim, shift_value=0.0)

    # problem = SphereFunction(dimension=args.dim, shift_value=0.0)
    AlgorithmClass = ALGORITHM_MAP[args.algorithm]

    print("======================================")
    print(f"Algorithm: {args.algorithm.upper()}")
    print(f"Function: {problem.__class__.__name__}")
    print(f"Dimension: {args.dim}")
    print("======================================")

    # ★追加：収束履歴を保存するフォルダを作成
    history_dir = f"results/histories/{args.algorithm}_{problem.__class__.__name__}_d{args.dim}"
    if not os.path.exists(history_dir):
        os.makedirs(history_dir)
    
    best_fitnesses = []
    results_data = []
    
    for r in range(args.runs):
        print(f"\nRun {r + 1}/{args.runs}")
        optimizer = AlgorithmClass(problem=problem)
        
        if args.algorithm == 'pso':
            # Java版のPSOApplicationで設定されていたパラメータ
            optimizer.set_parameters(inertia_weight=0.729, c1=1.49445, c2=1.49445)

        start_time = time.time()
        best_solution, best_fitness, fitness_history = optimizer.optimize()
        end_time = time.time()

        runtime_ms = (end_time - start_time) * 1000
        best_fitnesses.append(best_fitness)
        
        # ★追加：この実行結果を辞書として保存
        run_result = {
            'algorithm': args.algorithm,
            'run': r + 1,
            'dimension': args.dim,
            'final_fitness': best_fitness,
            'runtime_ms': runtime_ms
        }
        results_data.append(run_result)
        
        # ★追加：この回の収束履歴をCSVとして保存
        history_df = pd.DataFrame({'fitness': fitness_history})
        history_df.index.name = 'iteration'
        history_df.to_csv(f"{history_dir}/run_{r+1}.csv")

        print(f"Best fitness: {best_fitness}")
        print(f"History for run {r+1} saved.")
        print(f"Runtime: {runtime_ms:.2f} ms")

    mean_fitness = np.mean(best_fitnesses)
    std_fitness = np.std(best_fitnesses)

    print("\n======================================")
    print(f"Results Summary for {args.algorithm.upper()}")
    print(f"Mean Fitness: {mean_fitness}")
    print(f"Std Dev Fitness: {std_fitness}")
    print("======================================\n")
    
    # ★追加：全実行結果をCSVファイルに保存
    # resultsフォルダがなければ作成する
    if not os.path.exists('results'):
        os.makedirs('results')
        
    # pandasのDataFrameに変換してCSVとして出力
    df_results = pd.DataFrame(results_data)
    results_filename = f"results/{args.algorithm}_{problem.__class__.__name__}_d{args.dim}.csv"
    df_results.to_csv(results_filename, index=False)
    
    print(f"\nResults saved to {results_filename}")


if __name__ == "__main__":
    main()