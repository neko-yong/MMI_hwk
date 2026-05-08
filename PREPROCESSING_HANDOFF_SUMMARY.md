# 任务1：脑电数据预处理交接摘要

## 1. 当前完成情况

任务1预处理模块已经完成最终整理。当前流程可以从 DEAP 原始 `.bdf` 文件出发，自动恢复 trial，完成滤波、基线校正，并生成默认建模数据和官方 `.dat` 风格输出。

已验证全 32 个 subject 均可成功处理。

## 2. 核心文件

核心入口：

```text
src/preprocess.py
```

统一函数：

```python
from src.preprocess import preprocess_subject
```

最终验证脚本：

```text
src/validate_integrated_preprocessing.py
```

历史复核工具：

```text
src/dev_tools/
```

## 3. 推荐使用方式

默认建模使用：

```python
result = preprocess_subject(subject_id=1)
stimulus = result["baseline_corrected_stimulus"]
```

如果需要官方 `.dat` 风格结构：

```python
result = preprocess_subject(
    subject_id=1,
    output_official_like=True,
)
```

## 4. 输出 shape

默认输出：

```text
baseline_corrected_stimulus: (40, 32, 30720)
```

含义：40 trials，32 EEG channels，60 秒 stimulus，512 Hz。

official-like 输出：

```text
official_like_data: (40, 40, 8064)
official_like_labels: (40, 4)
official_like_baseline_corrected_eeg: (40, 32, 7680)
official_like_sampling_rate: 128
```

说明：`official_like_data` 是项目自建流程生成的官方风格结构，不是官方 `.dat` 的逐点复刻。

## 5. ICA 策略

ICA 已实现，但默认不自动删除 component：

```text
use_ica=False
ica_exclude_components=[]
```

原因是多被试 ICA 分析、人工复核、MNE full pipeline 和 subject-specific 清洗实验均未证明默认删除 component 能稳定改善结果。

后续如果要加强伪迹去除，必须 subject-specific 人工复核，不要全局删除同一个 component 编号。

## 6. 验证结果

最终验证结论：

```text
32/32 subjects 成功
baseline_corrected_stimulus 全部为 (40, 32, 30720)
official_like_baseline_corrected_eeg 全部为 (40, 32, 7680)
official_like_labels 全部为 (40, 4)
official_like_data 全部为 (40, 40, 8064)
```

最终报告目录：

```text
results/final_preprocessing/
```

## 7. 队友后续怎么接

如果做特征提取和分类，直接接：

```python
result["baseline_corrected_stimulus"]
```

如果需要和官方 `.dat` 对齐或做 128 Hz 建模，接：

```python
result["official_like_baseline_corrected_eeg"]
```

如果需要模拟官方 `.dat` 的数据结构，接：

```python
result["official_like_data"]
result["official_like_labels"]
```

一般情况下，队友不需要手动设置 ICA 删除成分。
