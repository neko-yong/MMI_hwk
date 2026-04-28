# DEAP EEG 数据预处理模块说明

## 1. 当前模块目标

本模块负责课程作业中的任务1：脑电信号预处理。当前目标是从 DEAP 原始 BDF 数据中恢复每个被试的 EEG trial，并生成后续特征提取和情感分类可直接使用的预处理结果。

当前预处理模块已经完成全 32 个 subject 的稳定性验证。

## 2. 数据放置位置

原始 BDF 数据应放在：

```text
data/DEAP/original/
```

元数据应放在：

```text
data/DEAP/metadata/
```

## 3. 核心文件

核心预处理文件是：

```text
src/preprocess.py
```

它负责 BDF 读取、事件识别、trial 切分、滤波、baseline correction 和可选 ICA。

开发验证和复核工具集中放在：

```text
src/dev_tools/
```

这些脚本用于历史验证、结果复核和调试，不是主流程必需。

后续建模相关文件包括：

```text
src/features.py
src/train_baseline.py
src/active_learning.py
```

这些文件属于特征提取、分类和主动学习模块，不是当前预处理任务的核心。

## 4. 预处理流程

当前预处理流程如下：

```text
BDF 读取
-> 事件通道识别
-> code 3/4/5 trial 恢复
-> baseline/stimulus 切分
-> 固定长度整理
-> 4-45 Hz bandpass
-> 50 Hz notch
-> baseline correction
-> 输出 baseline_corrected_stimulus
```

事件定义：

```text
code 3 = baseline start
code 4 = stimulus start
code 5 = stimulus end
baseline = [3, 4)
stimulus = [4, 5)
```

## 5. 输出数据形状

每个 subject 最终输出：

```text
baseline_corrected_stimulus shape = (40, 32, 30720)
```

含义：

```text
40 个 trial
32 个 EEG 通道
30720 个采样点，对应 60 秒 x 512 Hz
```

后续特征提取和建模应优先使用 `preprocess_subject()` 返回的 `baseline_corrected_stimulus`。

## 6. 事件通道处理说明

DEAP 原始 BDF 的事件通道在不同 subject 中存在结构差异：

```text
s01-s23 通常有 Status 通道
s24-s32 可能没有名为 Status 的通道，事件通道名可能为空
```

当前 `src/preprocess.py` 已支持：

```text
优先识别 Status 通道
如果没有 Status，则检查最后 1-2 个事件样通道
从冗余 code 3/4/5 中恢复合法 40 个 trial
按 3 -> 4 -> 5 顺序恢复 trial
过滤 baseline/stimulus 时长异常的候选 trial
```

因此，后 9 个 subject 中出现 code 3/5 冗余事件时，当前逻辑也可以恢复合法的 40 个 trial。

## 7. ICA 伪迹去除说明

ICA 已在 `src/preprocess.py` 中实现，可通过参数启用：

```python
preprocess_subject(
    subject_id=1,
    use_ica=True,
    ica_exclude_components=[]
)
```

但默认不自动删除 component。

当前默认策略是：

```text
use_ica=False
ica_exclude_components=[]
```

原因是多被试 ICA 分析、人工复核、MNE 复核和候选清洗实验均未证明“默认删除某些 component”能稳定改善数据。因此，保守策略是默认不删，避免误删有效 EEG 成分。

如果后续需要更强伪迹清洗，可以使用以下人工复核工具：

```text
src/dev_tools/review_ica_components.py
src/dev_tools/mne_ica_review.py
src/dev_tools/test_mne_ica_cleaning_candidates.py
```

注意：ICA component 删除必须是 subject-specific 的人工决策，不要全局删除同一个 component 编号。

## 8. 重要验证结果

最终预处理验证报告集中保存在：

```text
results/final_preprocessing/
```

重点结论：

```text
全 32 个 subject 预处理成功
全部成功输出 shape = (40, 32, 30720)
ICA 默认不删是经过验证后的保守策略，不是遗漏
```

建议优先查看：

```text
results/final_preprocessing/all_subjects_preprocessing_report.txt
results/final_preprocessing/all_subjects_preprocessing_summary.csv
results/final_preprocessing/ica_artifact_recommendation_report.txt
results/final_preprocessing/ica_manual_review_report.txt
results/final_preprocessing/mne_ica_review_report.txt
results/final_preprocessing/mne_ica_cleaning_test_report.txt
```

## 9. 常用运行命令

运行预处理 smoke test：

```bash
python -m src.preprocess
```

如果需要重新做全量验证：

```bash
python -m src.dev_tools.run_preprocessing_all_subjects
```

说明：`src/dev_tools/` 中的脚本主要用于历史复核和调试。如果某些脚本因为归档移动或路径变化导致无法直接运行，不影响核心 `src/preprocess.py` 的使用。

## 10. 给队友的使用建议

后续特征提取和建模直接调用：

```python
from src.preprocess import preprocess_subject

result = preprocess_subject(subject_id=1)
stimulus = result["baseline_corrected_stimulus"]
```

队友不需要手动设置 ICA 删除成分。

如果确实要做更强伪迹清洗，必须先做 subject-specific 人工复核，再决定 `ica_exclude_components`。不要把某个 component 编号作为所有 subject 的全局默认删除规则。
