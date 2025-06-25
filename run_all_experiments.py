import subprocess
import time
import argparse

# --- 利用可能な全てのアルゴリズムと問題をここで定義 ---
ALL_ALGORITHMS = [
    'pso',
    'efwa',
]

ALL_PROBLEMS = [
    'sphere',
    'rosenbrock',
]
# -------------------------------------------------

def main():
    # --- argparseを使って引数を設定 ---
    parser = argparse.ArgumentParser(description="Run a batch of optimization experiments.")
    parser.add_argument('--algorithms', nargs='+', choices=ALL_ALGORITHMS, 
                        help=f"Specify algorithms to run. If not provided, all will be run.")
    parser.add_argument('--problems', nargs='+', choices=ALL_PROBLEMS,
                        help=f"Specify problems to run. If not provided, all will be run.")
    
    # ★★★次元数と試行回数の引数を追加★★★
    # default値を設定しているので、指定しない場合はそれぞれ20と30が使われます。
    parser.add_argument('--dim', type=int, default=20, help='Dimension of the problem.')
    parser.add_argument('--runs', type=int, default=30, help='Number of runs for each experiment.')

    args = parser.parse_args()
    
    # --- 引数に基づいて、実行するリストを決定 ---
    algos_to_run = args.algorithms if args.algorithms else ALL_ALGORITHMS
    probs_to_run = args.problems if args.problems else ALL_PROBLEMS
    
    # -------------------------------------------------
    
    print("Starting batch execution...")
    print(f"Algorithms to run: {algos_to_run}")
    print(f"Problems to run: {probs_to_run}")
    print(f"Dimension for all: {args.dim}") # ★表示を追加
    print(f"Runs for each: {args.runs}")     # ★表示を追加
    
    start_time = time.time()
    
    total_experiments = len(algos_to_run) * len(probs_to_run)
    current_experiment = 0

    for algo in algos_to_run:
        for prob in probs_to_run:
            current_experiment += 1
            print("\n" + "="*70)
            print(f"Running Experiment {current_experiment}/{total_experiments}: Algorithm='{algo}', Problem='{prob}'")
            print("="*70)
            
            command = [
                'python',
                'run_experiment.py',
                '--algorithm', algo,
                '--problem', prob,
                '--dim', str(args.dim),   # ★ハードコードされていた値を引数に置き換え
                '--runs', str(args.runs)  # ★ハードコードされていた値を引数に置き換え
            ]
            
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