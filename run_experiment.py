import numpy as np
import time
import argparse
import pandas as pd # ★pandasをインポート
import os # ★osをインポート

# 作成したパッケージからクラスをインポート
from problems import SphereFunction, RosenbrockFunction, AckleyFunction
from algorithms import PSO,EFWA, DynFWA, HCFWA, SaFWA, SaHCFWA

# 実行可能なアルゴリズムを管理する辞書
ALGORITHM_MAP = {
    'pso': PSO,
    'efwa': EFWA,
    'dynfwa': DynFWA,
    'hcfwa': HCFWA,
    'safwa': SaFWA,
    'sahcfwa': SaHCFWA,
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
    # parser.add_argument('--problem', type=str, required=True, choices=PROBLEM_MAP.keys(), help='The problem function to optimize.')
    # choices=PROBLEM_MAP.keys() の部分を削除する
    parser.add_argument('--problem', type=str, required=True, help='The problem function to optimize (e.g., sphere, cec2017/f1).')
    parser.add_argument('--dim', type=int, default=20, help='Dimension of the problem.')
    parser.add_argument('--runs', type=int, default=5, help='Number of runs.')
    parser.add_argument('--no_shift', action='store_true', help='(CEC2017 only) Disable shift transformation')
    parser.add_argument('--no_rotate', action='store_true', help='(CEC2017 only) Disable rotation transformation')
    
    args = parser.parse_args()
    
    # ProblemClass = PROBLEM_MAP[args.problem]
    # problem = ProblemClass(dimension=args.dim, shift_value=0.0)

    # problem = SphereFunction(dimension=args.dim, shift_value=0.0)
    AlgorithmClass = ALGORITHM_MAP[args.algorithm]
    
    # === (ここから改造ブロック) ===
    import importlib # ★動的インポートのためのライブラリ
    from problems.base_problem import BaseProblem # ★共通ルール(BaseProblem)をインポート

    # PROBLEM_MAP = { ... } の行は削除（またはコメントアウト）してOKです

    try:
        problem_name_str = args.problem  # 例: 'sphere' または 'cec2017/f1'

        if '/' in problem_name_str:
            # --- パターンA: "スイート形式" (例: cec2017/f1) の場合 ---
            
            # 1. 'cec2017' と 'f1' に分割
            suite_name, func_name = problem_name_str.split('/')
            
            # 2. 読み込むファイル名を指定 (例: 'problems.cec2017')
            #    (problems/cec2017.py を指す)
            module_name = f"problems.{suite_name}"
            
            # 3. 読み込むクラス名を指定 (例: 'CEC2017_F1')
            class_name = f"{suite_name.upper()}_{func_name.upper()}"

        else:
            # --- パターンB: "単一ファイル形式" (例: sphere) の場合 ---
            
            # 1. 読み込むファイル名を指定 (例: 'problems.sphere')
            #    (problems/sphere.py を指す)
            module_name = f"problems.{problem_name_str}"
            
            # 2. 読み込むクラス名を指定 (例: 'SphereFunction')
            #    (これは既存のファイル規約に合わせる)
            if problem_name_str == 'sphere':
                class_name = 'SphereFunction'
            elif problem_name_str == 'rosenbrock':
                class_name = 'RosenbrockFunction'
            else:
                # 今後、他の単一ファイルを追加する場合の汎用ルール (例: 'my_func' -> 'My_funcFunction')
                class_name = problem_name_str.capitalize() + "Function"

        # 4. モジュール(ファイル)を動的にインポート
        # (例: 'problems.cec2017' または 'problems.sphere' を読み込む)
        module = importlib.import_module(module_name)
        
        # 5. モジュールからクラス(設計図)を取得
        # (例: 'CEC2017_F1' または 'SphereFunction' クラスを取り出す)
        ProblemClass = getattr(module, class_name)

        # 6. 安全チェック (BaseProblemを継承しているか)
        if not issubclass(ProblemClass, BaseProblem):
            raise TypeError(f"{class_name} does not inherit from BaseProblem")

        # 7. インスタンス化 (実体を作る)
        if 'cec2017' in problem_name_str:
            # CEC関数の場合は、フラグを渡す
            problem = ProblemClass(dimension=args.dim, 
                                 shift_flag=not args.no_shift, 
                                 rotate_flag=not args.no_rotate)
        else:
            # それ以外(sphereなど)の場合は、従来通り
            problem = ProblemClass(dimension=args.dim)

    except (ImportError, AttributeError, TypeError) as e:
        print(f"--- エラー: 問題 '{args.problem}' の読み込みに失敗しました ---")
        print(f"詳細: {e}")
        print("指定した問題名、ファイル構造、クラス名が規約通りか確認してください。")
        exit()
    # === (ここまで改造ブロック) ===

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