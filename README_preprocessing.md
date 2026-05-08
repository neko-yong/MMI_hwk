# DEAP EEG 数据预处理模块说明

## 1. 模块定位

本模块对应课程任务 1：脑电信号预处理。

当前项目从 DEAP 原始 BDF 文件出发，完成事件解析、trial 分段、滤波、基线校正，并提供接近官方 `.dat` 结构的 official-like 输出，供后续特征提取、情感分类建模和课程报告展示使用。

本模块的核心目标不是逐点复刻 DEAP 官方预处理文件，而是建立一条稳定、透明、可复现、适合课程作业展示的预处理流程。

## 2. 数据目录要求

原始 BDF 文件应放在：

```text
data/DEAP/original/
```

文件命名要求：

```text
s01.bdf
s02.bdf
...
s32.bdf
```

元数据和标签文件应放在：

```text
data/DEAP/metadata/
```

其中 `participant_ratings.csv` 用于生成 Valence / Arousal / Dominance / Liking 标签。

如果数据文件过大，仓库中可以不提交原始 BDF 或官方 `.dat` 大文件，但必须保持上述目录结构一致。

## 3. 核心入口

统一入口为：

```python
from src.preprocess import preprocess_subject
```

默认调用：

```python
result = preprocess_subject(subject_id=1)
```

官方风格输出调用：

```python
result = preprocess_subject(
    subject_id=1,
    output_official_like=True,
)
```

默认调用保持向后兼容，不会改变已有特征提取和建模代码的使用方式。

## 4. 默认输出说明

默认 `result` 中推荐关注以下字段：

- `filtered_stimulus`: 经过 4-45 Hz bandpass 和 50 Hz notch 后的 stimulus EEG。
- `baseline_corrected_stimulus`: 在滤波后，用对应 trial 的 baseline 均值校正后的 stimulus EEG。
- `preprocessing_info`: 仅在启用 official-like 输出时额外包含官方风格输出说明。

最推荐给后续特征提取和建模使用的是：

```python
stimulus = result["baseline_corrected_stimulus"]
```

该字段形状为：

```text
baseline_corrected_stimulus.shape == (40, 32, 30720)
```

含义：

- 40 个 trial
- 32 个 EEG 通道
- 30720 个采样点
- 对应 60 秒 x 512 Hz

## 5. 官方风格输出说明

当 `output_official_like=True` 时，`result` 会额外返回：

- `official_like_data`
- `official_like_labels`
- `official_like_baseline_corrected_eeg`
- `official_like_sampling_rate`
- `preprocessing_info`

已验证的输出形状为：

```text
official_like_data.shape == (40, 40, 8064)
official_like_labels.shape == (40, 4)
official_like_baseline_corrected_eeg.shape == (40, 32, 7680)
official_like_sampling_rate == 128
```

含义：

- `official_like_data` 是官方 `.dat` 风格结构。
- `official_like_data` 的语义为 40 trials x 40 channels x 8064 samples。
- `8064 = 63 秒 x 128 Hz`。
- 63 秒由前 3 秒 baseline 和后 60 秒 stimulus 组成。
- `official_like_labels` 的顺序为 Valence / Arousal / Dominance / Liking。
- `official_like_baseline_corrected_eeg` 是 32 个 EEG 通道、60 秒 stimulus、128 Hz 的 baseline-corrected 版本，更适合后续建模和官方 `.dat` 对比。

必须注意：`official_like_data` 是项目自建流程生成的官方风格数据，不是 DEAP 官方 `.dat` 的逐点复刻。它用于保持结构一致和方便交接，不应在报告中表述为“官方数据完全复现”。

## 6. 预处理流程

当前稳定预处理流程如下：

1. 读取 DEAP 原始 BDF。
2. 选择前 32 个 EEG 通道。
3. 自动识别事件通道。
4. 从 code 3 / 4 / 5 恢复 trial。
5. 切分 baseline 和 stimulus。
6. 固定长度整理。
7. 进行 4-45 Hz bandpass。
8. 进行 50 Hz notch。
9. 进行 baseline correction。
10. 可选生成 official-like 128 Hz 输出。

事件定义：

```text
code 3 = baseline start
code 4 = stimulus start
code 5 = stimulus end
baseline = [3, 4)
stimulus = [4, 5)
```

## 7. 事件通道鲁棒处理说明

DEAP 原始 BDF 文件中，不同 subject 的事件通道结构存在差异：

- `s01-s23` 通常有名为 `Status` 的事件通道。
- `s24-s32` 可能没有名为 `Status` 的通道，事件通道名可能为空。

当前代码的处理策略：

- 优先查找 `Status` 通道。
- 如果找不到 `Status`，则检查最后 1-2 个事件样通道。
- 从事件序列中寻找合法 `3 -> 4 -> 5` 模式。
- 支持处理冗余 code 3 / code 5 事件。
- 通过时长过滤恢复合法 40 个 trial。

时长过滤规则：

- baseline 合理范围：4-8 秒。
- stimulus 合理范围：55-65 秒。

该逻辑已解决后 9 个 subject 中事件通道命名不同和冗余事件导致的 trial 恢复问题。

## 8. ICA 伪迹去除说明

ICA 已在 `src/preprocess.py` 中实现，可以通过参数启用：

```python
result = preprocess_subject(
    subject_id=1,
    use_ica=True,
    ica_exclude_components=[0],
)
```

也可以只启用 ICA 但不删除成分：

```python
result = preprocess_subject(
    subject_id=1,
    use_ica=True,
    ica_exclude_components=[],
)
```

当前默认策略是不自动删除 ICA component：

```text
use_ica=False
ica_exclude_components=[]
```

原因是已有验证包括：

- 多被试 ICA 分析。
- 人工复核辅助系统。
- MNE full pipeline 复核。
- subject-specific 清洗实验。

这些结果均未证明自动删除某些 component 能稳定改善官方相似度、高频指标或整体信号质量。因此默认策略是保守不删，避免误删有效脑电成分。

不建议全局删除同一个 ICA component 编号。如果后续要做更强伪迹清洗，必须 subject-specific 人工复核，再决定 `ica_exclude_components`。

## 9. 常用运行命令

检查核心文件编译：

```bash
python -m py_compile src/preprocess.py src/validate_integrated_preprocessing.py
```

如果存在后续建模脚本，也可以一起检查：

```bash
python -m py_compile src/features.py src/train_baseline.py src/active_learning.py
```

运行单被试预处理 smoke test：

```bash
python -m src.preprocess
```

运行全 32 被试整合验证：

```bash
python -m src.validate_integrated_preprocessing
```

历史复核工具保存在：

```text
src/dev_tools/
```

这些脚本主要用于复核、调试和报告补充，不是主流程必需。

## 10. 验证结果摘要

当前最终验证结果：

- 32/32 subjects 成功。
- `baseline_corrected_stimulus` 全部为 `(40, 32, 30720)`。
- `official_like_baseline_corrected_eeg` 全部为 `(40, 32, 7680)`。
- `official_like_labels` 全部为 `(40, 4)`。
- `official_like_data` 全部为 `(40, 40, 8064)`。
- 默认 ICA 策略是不自动删除 component。

最终报告集中保存在：

```text
results/final_preprocessing/
```

其中包括全被试预处理验证、official-like 输出验证、ICA 复核、MNE 复核和官方 `.dat` 对比相关报告。

## 11. 给队友的使用建议

如果后续做特征提取和分类，优先使用：

```python
result["baseline_corrected_stimulus"]
```

如果要与官方 `.dat` 对齐或做 128 Hz 建模，使用：

```python
result = preprocess_subject(
    subject_id=1,
    output_official_like=True,
)

eeg_128hz = result["official_like_baseline_corrected_eeg"]
```

如果要模拟官方 `.dat` 结构，使用：

```python
official_like_data = result["official_like_data"]
official_like_labels = result["official_like_labels"]
```

队友不需要自己手动设置 ICA 删除成分。若后续确实要加强伪迹去除，应先做 subject-specific 人工复核，不要全局删除同一个 component 编号。

## 12. 注意事项

- 不要提交原始 `.bdf` 大文件，除非团队需要。
- 不要提交官方 `.dat` 大文件，除非团队需要。
- 不要随意修改 `src/preprocess.py` 的事件恢复逻辑。
- 不要把 MNE cleaned 作为默认策略。
- 不要把某个 ICA component 编号作为所有 subject 的默认删除规则。
- `official_like_data` 不是官方 `.dat` 的逐点复刻，只是项目自建流程生成的官方风格结构。
