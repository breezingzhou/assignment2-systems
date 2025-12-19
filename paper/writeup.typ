#set text(
  font: "Noto Sans SC:style=DemiLight",
  lang: "zh",
  size: 14pt
)
#show title: set align(center)
#show title: set text(size: 22pt)

#title("Assignment2 Writeup")


== Problem (benchmarking_script)
+ forward pass需要
+ backward pass不需要
+ 需要在benchmarking_script中加入

== Problem(nsys_profile):

== Problem(mixed_precision_accumulation)
 - 结果分别为：
    - tensor(10.0001)
    - tensor(9.9531, dtype=torch.float16)
    - tensor(10.0021)
    - tensor(10.0021)
 - 第一个结果是最精准的，误差来源于float32存储精度限制以及相加时的精度损失
 - 第二个结果误差最大，主要是因为float16精度较float32更低，累计误差较大
 - 第三、四结果相同，相比第二个实验，总误差大幅降低，是因为使用了float32保存结果，相加时的精度损失降低
