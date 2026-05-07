# 数据制作与训练执行顺序

本文档按当前仓库的真实链路整理：不要把 `all_*.pt` 当成初始数据。它们是中间产物，完整流程是：

```text
原始 benchmark 文件
  -> JSON
  -> split-field .pt 训练数据
  -> dataloader rounds 缓存
  -> main.py 训练
```

所有命令默认从项目根目录执行。换到新项目时，把下面路径替换成新项目根目录：

```bash
cd <NEW_PROJECT_ROOT>
```

## 0. 新项目最小复制清单

如果你只准备把原始数据复制到一个新项目，最少需要复制这三个大文件，并且文件名和位置要保持一致：

```bash
mkdir -p data/nasbench101 data/nasbench201

# NASBench-101 原始 benchmark
cp <OLD_PROJECT>/data/nasbench101/nasbench_full.tfrecord \
   data/nasbench101/nasbench_full.tfrecord

# NASBench-101 额外测量的 latency，注意它在 data/ 根目录，不在 data/nasbench101/ 下面
cp <OLD_PROJECT>/data/nasbench101/nasbench101_latency.csv \
   data/nasbench101/nasbench101_latency.csv

# NASBench-201 原始 API pth，当前脚本硬编码使用 v1_1 文件名
cp <OLD_PROJECT>/data/nasbench201/NAS-Bench-201-v1_1-096897.pth \
   data/nasbench201/NAS-Bench-201-v1_1-096897.pth
```

复制后先做一次路径检查：

```bash
test -f data/nasbench101/nasbench_full.tfrecord
test -f data/nasbench101/nasbench101_latency.csv
test -f data/nasbench201/NAS-Bench-201-v1_1-096897.pth
head -1 data/nasbench101/nasbench101_latency.csv
```

`head -1` 应该看到：

```text
hash,latency_mean_ms
```

不要把下面这些中间产物当成原始数据复制过去，除非你明确想复用旧处理结果：

```bash
data/nasbench101/nasbench101.json
data/nasbench201/nasbench201.json
data/nasbench101/all_nasbench101.*
data/nasbench201/all_nasbench201.*
data/nasbench101/rounds*/
data/nasbench201/rounds*/
results/
```

如果这些中间产物被一起复制过去，又想重新生成数据，先删掉：

```bash
rm -f data/nasbench101/nasbench101.json
rm -f data/nasbench201/nasbench201.json
rm -f data/nasbench101/all_nasbench101.*
rm -f data/nasbench201/all_nasbench201.*
rm -rf data/nasbench101/rounds*/
rm -rf data/nasbench201/rounds*/
```

数据生成相关代码至少需要保留：

```bash
preprocessing/gen_json_101.py
preprocessing/gen_json_201.py
preprocessing/nasbench/
preprocessing/nas_201_api/
generate_data.py
models/encoders/
```

训练还需要完整的 `main.py`、`config.py`、`models/`、`datasets/`、`training/`、`utils/`。

必须从项目根目录执行命令，例如 `python preprocessing/gen_json_101.py`。不要 `cd preprocessing` 后运行，因为脚本里的 `data/...` 路径是按项目根目录写死的。

## 1. 原始数据

先确认这些原始文件存在：

```bash
data/nasbench101/nasbench_full.tfrecord
data/nasbench101/nasbench101_latency.csv
data/nasbench201/NAS-Bench-201-v1_1-096897.pth
```

说明：

- `nasbench101` 的 TFRecord 只有架构、精度、训练时间等信息，不含推理 latency。
- `data/nasbench101/nasbench101_latency.csv` 是额外测出来并拼回去的 latency 文件，必须包含 `hash,latency_mean_ms` 两列。
- `nasbench201` 的 latency 来自 NAS-Bench-201 API 的 `get_latency(...)`。
- 两个入口脚本都有默认路径：`gen_json_101.py` 默认读 `data/nasbench101/nasbench_full.tfrecord` 和 `data/nasbench101/nasbench101_latency.csv`；`gen_json_201.py` 默认读 `data/nasbench201/NAS-Bench-201-v1_1-096897.pth`。如果新项目文件名不同，可以通过命令行参数覆盖。

## 2. 生成 JSON

第一步先把 benchmark 原始文件转换成统一 JSON。

```bash
python preprocessing/gen_json_101.py
python preprocessing/gen_json_201.py
```

产物：

```bash
data/nasbench101/nasbench101.json
data/nasbench201/nasbench201.json
```

这一步做的事情：

- `gen_json_101.py` 读取 `nasbench_full.tfrecord`，再用 `nasbench101_latency.csv` 按 hash 拼入 latency。
- `gen_json_201.py` 读取 `NAS-Bench-201-v1_1-096897.pth`，导出 accuracy、graph、training_time、latency。

如果 101 的 latency CSV 和 TFRecord hash 对不上，`gen_json_101.py` 会直接报错。这是合理的，不能跳过。

## 3. 生成训练用 split-field PT

第二步把 JSON 转成 dataloader 真正读取的 split-field `.pt`。

```bash
python generate_data.py --dataset all
```

如果只生成单个数据集，也可以指定：

```bash
python generate_data.py --dataset nasbench101
python generate_data.py --dataset nasbench201
```

主要产物：

```bash
data/nasbench101/all_nasbench101.meta.pt
data/nasbench101/all_nasbench101.ops.pt
data/nasbench101/all_nasbench101.code_rel_pos.pt
data/nasbench101/all_nasbench101.reachability.pt
data/nasbench101/all_nasbench101.dir_pe_ml.pt
data/nasbench101/all_nasbench101.dir_pe_ml2.pt
data/nasbench101/all_nasbench101.latency.pt

data/nasbench201/all_nasbench201.meta.pt
data/nasbench201/all_nasbench201.ops.pt
data/nasbench201/all_nasbench201.code_rel_pos.pt
data/nasbench201/all_nasbench201.reachability.pt
data/nasbench201/all_nasbench201.dir_pe_ml.pt
data/nasbench201/all_nasbench201.dir_pe_ml2.pt
data/nasbench201/all_nasbench201.latency.pt
```

这一步额外制作进去的字段很关键：

- `code / code_rel_pos / code_depth`：tokenizer 生成的结构编码。
- `op_depth / reachability / in_degree / out_degree`：图拓扑特征。`reachability` 是有向可达性，不再保存最短路径长度。
- `dir_pe_rw`：有向随机游走 PE。
- `dir_pe_ml`：q=0.25 的 Magnetic Laplacian PE，当前 `model56` 主模型会读它。
- `dir_pe_ml2`：q=0 的 Magnetic Laplacian PE，`model56_q0` 和 `model56_undirected_q0` 会读它。
- `latency`：如果 JSON 有 latency，会在这里做 min-max 归一化，并把反归一化参数写入 `*.meta.pt` 的 `norm_params`。

## 4. q=0 Magnetic PE

`generate_data.py` 会同时生成 `dir_pe_ml` 和 `dir_pe_ml2`，不需要额外补丁脚本。

## 5. 清理 dataloader 缓存

只要重新生成过 JSON、split-field PT、latency、PE 字段，就必须删掉旧 rounds 缓存：

```bash
rm -rf ./data/nasbench101/rounds*/
rm -rf ./data/nasbench201/rounds*/
```

原因：`NasbenchDataset` 会把 train/val/test 切分后的样本缓存到 `data/<dataset>/rounds<runid>/`。不清理的话，训练可能继续读旧字段或旧切分。

## 6. 训练前 smoke test

先跑一个 1 epoch 小测试，确认数据、模型注册、dataloader、loss 都能通：

```bash
python main.py \
  --model model56 \
  --dataset nasbench201 \
  --percent 156 \
  --rounds 1 \
  --epochs 1 \
  --print_freq 1 \
  --device cuda \
  --predict_target latency \
  --graph_readout att
```

如果跑 q0 模型，先确认已经执行过第 4 步，然后把 `--model` 换成：

```bash
--model model56_q0
```

或：

```bash
--model model56_undirected_q0
```

## 7. 正式训练

当前有效训练入口是 `main.py`。两个可直接用的主线脚本是：

```bash
bash run_ours201_latency.sh
bash run_ours101_latency.sh
```

`run_ours201_latency.sh` 当前跑：

```text
dataset=nasbench201
model=model56
percent=156 469 781 1563
rounds=3
predict_target=latency
```

`run_ours101_latency.sh` 当前只跑：

```text
dataset=nasbench101
model=model56
percent=4236
rounds=3
predict_target=latency
```

如果需要 NASBench101 的完整四档结果，把 `run_ours101_latency.sh` 中的：

```bash
PERCENTS="4236"
```

改为：

```bash
PERCENTS="100 172 424 4236"
```

再执行：

```bash
bash run_ours101_latency.sh
```

## 8. 后台运行

这个环境里普通 `nohup bash xxx.sh &` 可能会很快退出。长任务建议用：

```bash
mkdir -p nohup_logs
setsid nohup bash -c 'cd <NEW_PROJECT_ROOT> && exec bash run_ours101_latency.sh' \
  > nohup_logs/run_ours101_latency_$(date +%Y%m%d_%H%M%S).log 2>&1 < /dev/null &
```

然后确认进程和日志：

```bash
pgrep -af 'run_ours101_latency|python main.py'
tail -f nohup_logs/run_ours101_latency_*.log
```

`run_ours201_latency.sh` 同理替换脚本名。

## 9. 结果位置

正式训练结束后重点看：

```bash
results/nasbench101/model56/<timestamp>/
results/nasbench201/model56/<timestamp>/
```

每个实验目录里主要文件：

```bash
config.json
summary_all_runs.json
run.log
details/run0.log
checkpoints/best_model_run0.pt
history_run0.json
training_curve_run0.png
```

项目根目录还会追加总日志：

```bash
run.log
```

兼容旧流程的聚合指标会写到：

```bash
results/metrics/*.pkl
```

## 10. preprocessing 入口

`preprocessing/` 顶层只保留两个主入口：

```bash
preprocessing/gen_json_101.py
preprocessing/gen_json_201.py
```

NASBench-101 的 latency CSV 拼接、hash 校验和可选字段诊断都在 `gen_json_101.py` 内完成；NASBench-201 的 latency 从 API 直接导出。split-field 字段、`reachability`、`dir_pe_ml(q=0.25)`、`dir_pe_ml2(q=0)` 统一由 `generate_data.py` 生成。

这些脚本当前不是可靠主入口，因为很多还在调用已经不存在的 `Experiment.py`：

```bash
run_ours101.sh
run_ours201.sh
run_ablation_study.sh
run_hparam_search.sh
scripts/run_*.sh
```

如果要继续用它们，需要先把 `python Experiment.py` 改成 `python main.py`，并检查参数是否仍然兼容。

## 11. 最终主线清单

从零跑 NASBench latency 的完整顺序：

```bash
cd <NEW_PROJECT_ROOT>

# 1. 原始 benchmark -> JSON
python preprocessing/gen_json_101.py
python preprocessing/gen_json_201.py

# 2. JSON -> split-field PT，并生成结构特征、PE、归一化 latency
python generate_data.py --dataset all

# 3. 清理旧 dataloader cache
rm -rf ./data/nasbench101/rounds*/
rm -rf ./data/nasbench201/rounds*/

# 4. smoke test
python main.py --model model56 --dataset nasbench201 --percent 156 --rounds 1 --epochs 1 --print_freq 1 --device cuda --predict_target latency --graph_readout att

# 5. 正式训练
bash run_ours201_latency.sh
bash run_ours101_latency.sh
```
