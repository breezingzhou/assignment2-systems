#set text(
  font: "Noto Sans SC:style=DemiLight",
  lang: "zh",
)
#show title: set align(center)
#show title: set text(size: 22pt)

#let frame(stroke) = (x, y) => (
  left: if x > 0 { 0pt } else { stroke },
  right: stroke,
  top: if y < 2 { stroke } else { 0pt },
  bottom: stroke,
)

#let frame_benchmarking(stroke) = (x, y) => (
  left: if x > 0 { 0pt } else { stroke },
  right: stroke,
  top: if y == 0 or y == 2 { stroke } else { 0pt },
  bottom: stroke,
)
#show table.cell.where(y: 0): set text(weight: "medium")


#title("Assignment2 Writeup")

== Problem (benchmarking_script)
+ cs336_systems/benchmark.py
+ - #table(
      columns: (auto, auto, auto, auto, auto),
      rows: 3,
      inset: (x: 10pt, y: 6pt),
      align: center,
      stroke: frame_benchmarking(1pt + rgb("21222C")),
      fill: (_, y) => if calc.odd(y) { rgb("EAF2F5") },

      [dataset], table.cell(colspan: 2, [forward]), table.cell(colspan: 2, [forward + backword]),
      [warmup=5,trial=10], [mean], [std], [mean], [std],
      [small], [0.0419], [0.0062], [0.1209], [0.0114],
      [medium], [0.0933], [0.0109], [0.2571], [0.0091],
      [large], [0.1695], [0.0189], [29.7924], [2.6315],
      [xl], [25.6897], [2.1010], [-], [-],
      [2.7B], [-], [-], [-], [-],
    )
  - 在small和medium模型上，forward和backward的耗时方差都很小
  - 在large模型上，forward的耗时方差依然较小，但backward的耗时方差变得很大
    - 可能是因为large模型的显存占用已经接近GPU的极限，导致在训练过程中出现了显存碎片化的问题，从而影响了backward的计算效率
    - 另外，large模型的backward计算量也更大，可能会受到其他系统资源（如内存带宽、计算单元利用率等）的影响，从而导致耗时的不稳定
  - 在xl模型上，forward的耗时方差变得非常大，backward的耗时无法测量
    - 可能是因为xl模型的显存占用已经超过了GPU的极限，导致在训练过程中频繁发生显存交换现象，从而极大地影响了forward的计算效率
+ - #table(
      columns: (auto, auto, auto, auto, auto),
      rows: 3,
      inset: (x: 10pt, y: 6pt),
      align: center,
      stroke: frame_benchmarking(1pt + rgb("21222C")),
      fill: (_, y) => if calc.odd(y) { rgb("EAF2F5") },

      [dataset], table.cell(colspan: 2, [forward]), table.cell(colspan: 2, [forward + backword]),
      [warmup=0,trial=10], [mean], [std], [mean], [std],
      [small], [0.1066], [0.1420], [0.1741], [0.1630],
      [medium], [0.1081], [0.0209], [0.2735], [0.0205],
      [large], [0.1765], [0.0455], [27.3891], [1.6979],
      [xl], [27.1398], [2.6152], [-], [-],
      [2.7B], [-], [-], [-], [-],
    )
  - #table(
      columns: (auto, auto, auto, auto, auto),
      rows: 3,
      inset: (x: 10pt, y: 6pt),
      align: center,
      stroke: frame_benchmarking(1pt + rgb("21222C")),
      fill: (_, y) => if calc.odd(y) { rgb("EAF2F5") },

      [dataset], table.cell(colspan: 2, [forward]), table.cell(colspan: 2, [forward + backword]),
      [warmup=2,trial=10], [mean], [std], [mean], [std],
      [small], [0.0487], [0.0066], [0.1014], [0.0072],
      [medium], [0.1081], [0.0171], [0.2525], [0.0090],
      [large], [0.1626], [0.0086], [25.6130], [1.1563],
      [xl], [17.8838], [0.5077], [-], [-],
      [2.7B], [-], [-], [-], [-],
    )
  - 在没有warmup的情况下，模型的forward和backward耗时均值及方差都较大，可能原因有：
    - 模型权重从 CPU 内存加载到 GPU 显存
    - CUDA 内核的编译与加载
    - 缓存初始化

  - 经过2次warmup后，模型的forward和backward耗时均值及方差显著降低
    - 说明warmup能够有效预热GPU的计算单元和内存带宽，从而提高计算效率的稳定性





== Problem(nsys_profile):

== Problem(mixed_precision_accumulation)
- 结果分别为：
  - tensor(10.0001)
  - tensor(9.9531, dtype=torch.float16)
  - tensor(10.0021)
  - tensor(10.0021)
- 第一个结果是最精准的，误差来源于float32存储精度限制以及相加时的精度损失
- 第二个结果误差最大，主要是因为float16精度较float32更低，累计误差更大
- 第三、四结果相同，相比第二个实验，总误差大幅降低，是因为使用了float32保存结果，相加时的精度损失降低

== Problem (benchmarking_mixed_precision)
+ #table(
    columns: (auto, auto),
    rows: 5,
    inset: (x: 20pt, y: 8pt),
    align: center,

    [数据], [类型],
    [autocast上下文中的模型参数], [FP32],
    [fc1层输出的结果], [FP16],
    [ln层输出的结果], [FP32],
    [模型预测的logits], [FP16],
    [loss], [FP32],
    [模型的梯度], [FP32],
  )
+
  - 均值、方差和归一化计算都对混合精度敏感
    - 均值计算的精度丢失
    - 方差计算的误差放大
    - 归一化的分母下溢
  - 如果使用BF16，可以不特殊处理norm层。BF16有更大的动态范围，能更好地表示极小值，减少下溢风险
+
  - 1
  - 1

== Problem(memory_profiling)


== Problem(pytorch_attention)

== Problem (torch_compile)

== Problem (flash_forward)
=== Note:
+ `flash attention`在实现的时候`q``k``v`需要考虑`batch_size`, 不能把`batch_size`与`seq_length`合并成一个维度来处理
+ 多用`einsum`
+ Grid 布局 $["Batch" dot M dot N]$
  - #table(

      columns: (1fr, 1fr, 3fr),
      rows: 4,
      inset: (x: 20pt, y: 8pt),
      align: center + horizon,

      [目标], [Grid 顺序], [理由],
      [最大化 Batch 间并行], [$("Batch", M)$], [让不同的 Batch 分布在不同的 SM 上，适合 Batch 很大、单矩阵很小的情况。],
      [最大化 L2 缓存利用],
      [$(M, "Batch")$],
      [将同一个 Batch 的不同 Tile 放在连续的 pid 中。由于 GPU 调度倾向于先执行连续的 pid，这能让这些 Program 共享 L2 里的 Batch 数据。],

      [避免显存碎片访问], [(最快变化轴, 其他)], [永远让 tl.program_id(0) 对应你内存中地址最连续的那个逻辑维度。],
    )
