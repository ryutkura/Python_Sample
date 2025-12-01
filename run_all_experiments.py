import sys
import subprocess
import time
import argparse
import os
import glob
import importlib
import re
from problems.base_problem import BaseProblem

# --- 利用可能な全てのアルゴリズムをここで定義 ---
ALL_ALGORITHMS = [
    'pso',
    'efwa',
    'dynfwa',
    'hcfwa',
    'safwa',
    'sahcfwa',
]

# --- problemsフォルダを自動スキャンして問題リストを作成する関数 ---
def get_all_problems():
    problem_list = []
    problems_dir = 'problems'
    for f_path in glob.glob(os.path.join(problems_dir, '*.py')):
        filename = os.path.basename(f_path)
        if filename not in ['__init__.py', 'base_problem.py']:
            problem_name = filename.replace('.py', '')
            if 'cec' in problem_name.lower():
                try:
                    module = importlib.import_module(f'problems.{problem_name}')
                    cec_func_names = [attr for attr in dir(module) if re.match(r'^CEC\d+_F\d+$', attr)]
                    for class_name in cec_func_names:
                        parts = class_name.split('_')
                        func_name = parts[1].lower()
                        suite_name_lower = parts[0].lower()
                        problem_list.append(f'{suite_name_lower}/{func_name}')
                except ImportError:
                    print(f"Warning: Could not import problem suite 'problems.{problem_name}'.")
                except Exception as e:
                    print(f"Warning: Error scanning {problem_name}.py: {e}")
            else:
                problem_list.append(problem_name)
    return sorted(list(set(problem_list)))

ALL_PROBLEMS = get_all_problems()
# -------------------------------------------------

def main():
    # --- argparseを使って引数を設定 ---
    parser = argparse.ArgumentParser(description="Run a batch of optimization experiments.")
    parser.add_argument('--algorithms', nargs='+', choices=ALL_ALGORITHMS, 
                        help=f"Specify algorithms to run. If not provided, all will be run.")
    parser.add_argument('--problems', nargs='+', choices=ALL_PROBLEMS,
                        help=f"Specify problems to run (e.g., sphere cec2017/f1).")
    parser.add_argument('--problem_folder', type=str, 
                        help='Specify a problem suite to run (e.g., cec2017).')

    # ★(改造点 1) 引数名を --dims (複数形) に変更し、nargs='+' を追加
    parser.add_argument('--dims', nargs='+', type=int, default=[20], 
                        help='One or more dimensions to run (e.g., --dims 10 30 50).')
    
    parser.add_argument('--runs', type=int, default=30, help='Number of runs for each experiment.')
    parser.add_argument('--no_shift', action='store_true', help='(CEC2017 only) Disable shift transformation')
    parser.add_argument('--no_rotate', action='store_true', help='(CEC2017 only) Disable rotation transformation')

    args = parser.parse_args()
    
    # --- 引数に基づいて、実行するリストを決定 ---
    algos_to_run = args.algorithms if args.algorithms else ALL_ALGORITHMS
    dims_to_run = args.dims  # ★(改造点 2) 次元数のリストを取得
    
    if args.problem_folder:
        folder_prefix = f"{args.problem_folder.lower()}/" 
        probs_to_run = [p for p in ALL_PROBLEMS if p.startswith(folder_prefix)]
        if not probs_to_run:
            print(f"Error: No problems found for folder/suite '{args.problem_folder}'.")
            return
    elif args.problems:
        probs_to_run = args.problems
    else:
        probs_to_run = ALL_PROBLEMS
    
    # -------------------------------------------------
    
    print("Starting batch execution...")
    print(f"Algorithms to run: {algos_to_run}")
    print(f"Problems to run: ({len(probs_to_run)} problems)")
    print(probs_to_run)
    print(f"Dimensions to run: {dims_to_run}") # ★(改造点 2) 表示を変更
    print(f"Runs for each: {args.runs}")
    
    start_time = time.time()
    
    total_experiments = len(algos_to_run) * len(probs_to_run) * len(dims_to_run) # ★(改造点 2)
    current_experiment = 0

    # ★(改造点 3) ループを3重にする (algo, prob, dim)
    for algo in algos_to_run:
        for prob in probs_to_run:
            for dim in dims_to_run: # ★(改造点 3) 次元数のループを追加
                current_experiment += 1
                print("\n" + "="*70)
                print(f"Running Experiment {current_experiment}/{total_experiments}:")
                print(f"  Algorithm: '{algo}'")
                print(f"  Problem:   '{prob}'")
                print(f"  Dimension: {dim}") # ★(改造点 3)
                print("="*70)
                
                command = [
                    sys.executable,
                    # 'python',
                    'run_experiment.py',
                    '--algorithm', algo,
                    '--problem', prob,
                    '--dim', str(dim),   # ★(改造点 3) args.dim をループ変数 dim に変更
                    '--runs', str(args.runs)
                ]
                
                if args.no_shift:
                    command.append('--no_shift')
                if args.no_rotate:
                    command.append('--no_rotate')
                
                try:
                    subprocess.run(command, check=True)
                except Exception as e:
                    print(f"!!! An error occurred during experiment: {algo} on {prob} (Dim {dim}). Halting batch run. !!!")
                    print(f"Error: {e}")
                    return

    end_time = time.time()
    print("\n" + "="*70)
    print("Batch execution completed successfully!")
    print(f"Total execution time: {(end_time - start_time) / 60:.2f} minutes")
    print("="*70)


if __name__ == "__main__":
    main()