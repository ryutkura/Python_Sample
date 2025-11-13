import subprocess
import time
import argparse
import os
import glob         # ★ファイル検索ライブラリ
import importlib  # ★動的インポートライブラリ
import re         # ★クラス名検索ライブラリ (正規表現)
from problems.base_problem import BaseProblem # ★BaseProblemをインポート

# --- 利用可能な全てのアルゴリズムをここで定義 ---
ALL_ALGORITHMS = [
    'pso',
    'efwa',
    'dynfwa',
    'hcfwa',
]

# ★(改造点 1) problemsフォルダを自動スキャンして問題リストを作成する関数
def get_all_problems():
    """
    'problems' フォルダをスキャンし、run_experiment.py が実行可能な
    問題名のリスト (例: 'sphere', 'cec2017/f1') を自動生成する。
    """
    problem_list = []
    problems_dir = 'problems'

    # 1. problems/ 直下の .py ファイル (sphere.py, rosenbrock.py など) をスキャン
    for f_path in glob.glob(os.path.join(problems_dir, '*.py')):
        filename = os.path.basename(f_path)
        if filename not in ['__init__.py', 'base_problem.py']:
            problem_name = filename.replace('.py', '')
            
            # cec2017.py のような "スイート・ファイル" の中身もスキャン
            if 'cec' in problem_name.lower():
                try:
                    # (例: 'problems.cec2017' モジュールをインポート)
                    module = importlib.import_module(f'problems.{problem_name}')
                    
                    # (例: 'CEC2017_F' + 数字 というパターンのクラス名をすべて検索)
                    cec_func_names = [attr for attr in dir(module) if re.match(r'^CEC\d+_F\d+$', attr)]
                    
                    for class_name in cec_func_names:
                        # 'CEC2017_F1' -> 'cec2017/f1'
                        parts = class_name.split('_')
                        func_name = parts[1].lower() # 'f1'
                        suite_name_lower = parts[0].lower() # 'cec2017'
                        problem_list.append(f'{suite_name_lower}/{func_name}')
                        
                except ImportError:
                    print(f"Warning: Could not import problem suite 'problems.{problem_name}'.")
                except Exception as e:
                    print(f"Warning: Error scanning {problem_name}.py: {e}")
            else:
                # 'sphere' などの通常の単一ファイル問題
                problem_list.append(problem_name)

    return sorted(list(set(problem_list))) # 重複を削除してソート

# ★(改造点 1) 手書きのリストの代わりに、自動スキャン関数を呼び出す
ALL_PROBLEMS = get_all_problems()
# -------------------------------------------------

def main():
    # --- argparseを使って引数を設定 ---
    parser = argparse.ArgumentParser(description="Run a batch of optimization experiments.")
    parser.add_argument('--algorithms', nargs='+', choices=ALL_ALGORITHMS, 
                        help=f"Specify algorithms to run. If not provided, all will be run.")
    
    # ★(改造点 1) choices を自動生成されたリストに変更
    parser.add_argument('--problems', nargs='+', choices=ALL_PROBLEMS,
                        help=f"Specify problems to run (e.g., sphere cec2017/f1).")
    
    # ★(改造点 2) スイート(フォルダ)指定で実行するための新しい引数を追加
    parser.add_argument('--problem_folder', type=str, 
                        help='Specify a problem suite to run (e.g., cec2017).')

    # ★★★次元数と試行回数の引数は変更なし★★★
    parser.add_argument('--dim', type=int, default=20, help='Dimension of the problem.')
    parser.add_argument('--runs', type=int, default=30, help='Number of runs for each experiment.')
    # ... parser.add_argument('--runs', ...) の次の行に追加 ...
    parser.add_argument('--no_shift', action='store_true', help='(CEC2017 only) Disable shift transformation')
    parser.add_argument('--no_rotate', action='store_true', help='(CEC2017 only) Disable rotation transformation')

    args = parser.parse_args()
    
    # --- 引数に基づいて、実行するリストを決定 ---
    algos_to_run = args.algorithms if args.algorithms else ALL_ALGORITHMS
    
    # ★(改造点 2) --problem_folder が指定された場合のロジック
    if args.problem_folder:
        # (例: 'cec2017' が指定されたら、'cec2017/' で始まる問題だけを抜き出す)
        folder_prefix = f"{args.problem_folder.lower()}/" 
        probs_to_run = [p for p in ALL_PROBLEMS if p.startswith(folder_prefix)]
        if not probs_to_run:
            print(f"Error: No problems found for folder/suite '{args.problem_folder}'.")
            print(f"Available problems: {ALL_PROBLEMS}")
            return
    elif args.problems:
        # (従来通り --problems が指定された場合)
        probs_to_run = args.problems
    else:
        # (何も指定されない場合は、ALL_PROBLEMS リスト全体を実行)
        probs_to_run = ALL_PROBLEMS
    
    # -------------------------------------------------
    
    print("Starting batch execution...")
    print(f"Algorithms to run: {algos_to_run}")
    print(f"Problems to run: ({len(probs_to_run)} problems)")
    print(probs_to_run) # 実行対象が多すぎる可能性があるのでリストで表示
    print(f"Dimension for all: {args.dim}")
    print(f"Runs for each: {args.runs}")
    
    start_time = time.time()
    
    total_experiments = len(algos_to_run) * len(probs_to_run)
    current_experiment = 0

    # このforループは、上で決定された probs_to_run リストの中身だけを実行する
    for algo in algos_to_run:
        for prob in probs_to_run:
            current_experiment += 1
            print("\n" + "="*70)
            print(f"Running Experiment {current_experiment}/{total_experiments}: Algorithm='{algo}', Problem='{prob}'")
            print("="*70)
            
            command = [
                'python',
                'run_experiment.py', # 呼び出すスクリプトは変更なし
                '--algorithm', algo,
                '--problem', prob,   # ここに 'sphere' や 'cec2017/f1' が入る
                '--dim', str(args.dim),
                '--runs', str(args.runs)
            ]
            
            # ★(改造点 3) シフト/回転オプションを run_experiment.py に引き継ぐ
            if args.no_shift:
                command.append('--no_shift')
            if args.no_rotate:
                command.append('--no_rotate')
            
            try:
                subprocess.run(command, check=True)
            except Exception as e:
                print(f"!!! An error occurred during experiment: {algo} on {prob}. Halting batch run. !!!")
                print(f"Error: {e}")
                return

    end_time = time.time()
    print("\n" + "="*70)
    print("Batch execution completed successfully!")
    print(f"Total execution time: {(end_time - start_time) / 60:.2f} minutes")
    print("="*70)


if __name__ == "__main__":
    main()