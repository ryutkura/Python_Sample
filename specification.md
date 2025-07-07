はい、承知いたしました。
他のAIにこのプロジェクトの構造を理解させ、新しいアルゴリズムを正しく追加させるための「仕様書」ですね。これまでの対話で構築してきた我々のシステムの本質を的確に伝える、プログラマー向けの明確な指示を作成します。

-----

## **新しい最適化アルゴリズム追加のための仕様書**

### 1\. プロジェクト概要

このプロジェクトは、様々な最適化アルゴリズムの性能を、共通のベンチマーク問題と実験フレームワーク上で比較・評価することを目的としています。

中心的な役割を果たすのは、以下の2つのスクリプトです。

  * `run_experiment.py`: **単一の実験**（特定のアルゴリズムと問題の組み合わせ）を実行する。
  * `run_all_experiments.py`: 複数の実験を**一括で実行**する。

新しく追加されるアルゴリズムは、これらのスクリプトから呼び出せるように、本仕様書で定義されたインターフェースに準拠する必要があります。

-----

### 2\. 満たすべき要件（インターフェース定義）

新しい最適化アルゴリズムは、以下の要件を**必ず**満たすクラスとして実装してください。

1.  **コンストラクタ (`__init__`)**:

      * 第一引数として、最適化対象となる\*\*`problem`オブジェクト\*\*を必ず受け取ります。
      * この`problem`オブジェクトは、`problems.BaseProblem`クラスのインスタンスであり、以下のプロパティとメソッドを提供します。
          * `problem.dimension`: 問題の次元数 (`int`)
          * `problem.lower_bounds`: 各次元の下限値 (`np.ndarray`)
          * `problem.upper_bounds`: 各次元の上限値 (`np.ndarray`)
          * `problem.evaluate(position)`: 解候補（`np.ndarray`）の評価値を計算するメソッド (`float`)
      * その他、アルゴリズムが必要とするパラメータ（個体数など）を引数として定義できます。

2.  **最適化メソッド (`optimize`)**:

      * アルゴリズムのメインループを実装する\*\*`optimize`\*\*という名前のメソッドを必ず持ちます。
      * このメソッドは引数を取りません。
      * 最適化が完了した後、以下の3つの要素を**タプル**として**必ず**返却（`return`）します。
        1.  `best_solution`: 最良解の位置ベクトル (`np.ndarray`)
        2.  `best_fitness`: 最良解の評価値 (`float`)
        3.  `fitness_history`: 各イテレーション（または世代）ごとの最良評価値の履歴 (`List[float]`)

-----

### 3\. 実装手順

新しいアルゴリズムを追加するには、以下の手順に従ってください。

#### **ステップ1: アルゴリズムファイルの作成**

`algorithms/`ディレクトリ内に、アルゴリズム名を冠した新しいPythonファイル（例: `new_algo.py`）を作成します。

#### **ステップ2: クラスの実装**

作成したファイル内に、上記の「2. 満たすべき要件」で定義されたインターフェースに準拠したクラスを実装します。以下のコードテンプレートを参考にしてください。

#### **ステップ3: `__init__.py`への登録**

`algorithms/__init__.py`ファイルを開き、新しく作成したクラスをインポートする行を追加します。これにより、プロジェクト全体から新しいアルゴリズムが認識されるようになります。

```python
# algorithms/__init__.py

from .pso import PSO
from .efwa import EFWA
from .dynfwa import DynFWA
from .hcfwa import HCFWA
from .new_algo import NewAlgo # <<< このように一行追加
```

#### **ステップ4: 実行スクリプトへの登録**

`run_experiment.py`と`run_all_experiments.py`の2つのファイルで、新しいアルゴリズムを**実行可能なリスト**に登録します。

  * **`run_experiment.py`**

    ```python
    # ...
    from algorithms import PSO, EFWA, DynFWA, HCFWA, NewAlgo # 1. インポートを追加

    ALGORITHM_MAP = {
        'pso': PSO,
        'efwa': EFWA,
        'dynfwa': DynFWA,
        'hcfwa': HCFWA,
        'new_algo': NewAlgo, # 2. 辞書にキーとクラスを追加
    }
    # ...
    ```

  * **`run_all_experiments.py`**

    ```python
    # ...
    ALL_ALGORITHMS = [
        'pso',
        'efwa',
        'dynfwa',
        'hcfwa',
        'new_algo', # ここにキーを追加
    ]
    # ...
    ```

-----

### 4\. コードテンプレート

新しいアルゴリズムファイル (`new_algo.py`) を作成する際は、以下のテンプレートをコピーして使用してください。`# ---`で区切られた部分に、アルゴリズム固有のロジックを実装します。

```python
# algorithms/new_algo.py

import numpy as np
from typing import List, Tuple
from problems import BaseProblem

class NewAlgo:
    """
    新しいアルゴリズムのクラス。
    この説明文（docstring）も適切に記述すること。
    """
    def __init__(self, 
                 problem: BaseProblem,
                 population_size: int = 50, # 例：アルゴリズム固有のパラメータ
                 max_iterations: int = 1000): # 例：アルゴリズム固有のパラメータ
        
        # 実験システムから渡されるproblemオブジェクトを保持
        self.problem = problem
        
        # アルゴリズム固有のパラメータを保持
        self.population_size = population_size
        self.max_iterations = max_iterations

        # 乱数生成器の準備
        self.rng = np.random.default_rng()

    def optimize(self) -> Tuple[np.ndarray, float, List[float]]:
        """
        最適化のメインループ。
        最終的に(best_solution, best_fitness, fitness_history)を返す。
        """
        # --- 1. 初期化処理 ---
        # 問題の次元数や境界をproblemオブジェクトから取得
        dim = self.problem.dimension
        lower_bounds = self.problem.lower_bounds
        upper_bounds = self.problem.upper_bounds
        
        # 最良解と評価値、履歴リストの初期化
        best_solution = np.zeros(dim)
        best_fitness = float('inf')
        fitness_history = []
        
        # 例：初期個体群の生成
        # population = self.rng.uniform(lower_bounds, upper_bounds, size=(self.population_size, dim))

        # --------------------

        # --- 2. メインループ ---
        for iteration in range(self.max_iterations):
            
            # --- ここにアルゴリズムの1世代分の処理を記述 ---
            #
            # 例：個体の評価と最良解の更新
            # current_fitnesses = np.array([self.problem.evaluate(ind) for ind in population])
            # current_best_idx = np.argmin(current_fitnesses)
            # if current_fitnesses[current_best_idx] < best_fitness:
            #     best_fitness = current_fitnesses[current_best_idx]
            #     best_solution = population[current_best_idx].copy()
            #
            # 例：個体の更新処理
            # ...
            # ---------------------------------------------
            
            # 毎世代の最良評価値を履歴に追加
            fitness_history.append(best_fitness)

        # --------------------

        # --- 3. 結果の返却 ---
        # 仕様書で定められた3つの値をタプルとして返す
        return best_solution, best_fitness, fitness_history
        # --------------------
```

-----

### 5\. 動作確認方法

上記の手順がすべて完了したら、以下のコマンドをターミナルで実行し、新しいアルゴリズムがエラーなく動作することを確認してください。

```bash
python run_experiment.py --algorithm new_algo --problem sphere --dim 10 --runs 3
```

実験が正常に完了し、結果が出力されれば、システムへの統合は成功です。