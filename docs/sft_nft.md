# MDRL 独立 SFT → NFT

## 隔离与兼容边界

本功能只新增 mdrl_finetune、configs/sft_nft 和相应测试。
models、utils、scripts、RL_utils、原有配置与原有测试均不修改。
原生预训练和原生种群微调继续使用原入口；新方法必须显式运行
python -m mdrl_finetune，不能通过原生训练入口意外启用。

新方法是无条件生成：三组靶点各训练独立模型，受体只进入筛选与奖励。
基础权重必须确认为原生预训练 iteration=985000；不会加载某轮原生 RL 权重冒充基础模型。
原生配置从检查点读取并严格加载权重，YAML 中的 architecture 只是约束，
不是伪造或替换检查点架构。要求原生离散类别扩散、1000步、七元素及原子/边 mask。

新检查点是独立格式 mdrl-sft-nft-v1，native_config 保存原生架构，
config 保存微调设置；不能把它当作原生预训练的完整 resume 检查点。
原有检查点格式和原生入口的处理没有改变。

## 入口与配置

三组目标各有 SFT/NFT 两份 YAML，位于 configs/sft_nft。
配置中的服务器绝对路径是已有记录，不代表已在服务器验证存在。
预训练路径中的 last.pt 可能继续变化；运行时会严格检查985000步并记录文件SHA256，
不得用其他训练步数冒充。建议正式实验前自行保留不可变的源权重副本并修改配置。

仅检查配置、不加载模型、不生成数据：

    python -B -m mdrl_finetune sft --config configs/sft_nft/mdrl-sft-brd4-cdk2-s2024-r1.yaml --schema-only

真实输入预检：在拥有原生MDRL依赖、权重与评分资产的环境执行，
会在CPU严格构建模型、加载权重和检查评分资产，但不生成、训练、对接或创建实验目录：

    python -B -m mdrl_finetune sft --config configs/sft_nft/mdrl-sft-brd4-cdk2-s2024-r1.yaml --validate-only

训练入口（这里只说明用法，不表示已经执行）：

    python -B -m mdrl_finetune sft --config CONFIG.yaml --output NEW_SFT_DIRECTORY
    python -B -m mdrl_finetune nft --config CONFIG.yaml --output NEW_NFT_DIRECTORY
    python -B -m mdrl_finetune nft --config CONFIG.yaml --init-checkpoint SFT8_FINAL.pt --output NEW_NFT_DIRECTORY

NFT配置默认指向对应的独立SFT目录：
experiment_data/mdrl/sft_nft/server_runs/mdrl-sft8-<pair>-s2024-r1/final.pt。
实际SFT输出路径应与它一致，或明确使用 --init-checkpoint 覆盖。
三组为 brd4-cdk2、thrombin-trypsin、chk1-auroraa。
新包不依赖FlowMol代码；原生MDRL模型依赖与OpenBabel、Vina、RDKit SA、
SFT阶段的XGBoost资产需要由使用者环境提供，不自动安装。

## SFT

默认8轮，round_00至round_07；不是8个梯度更新或8个数据集epoch。

每轮：生成 → 评分 → 合并完整历史成功池、按isomeric SMILES去重
（最新记录覆盖，平分按初次插入顺序）→ Top4000 → 去噪训练 → 保存。
第一轮3000次尝试，其余每轮2000次，不补采成功分子。
完整历史池与训练用Top4000分别保存，不只保存精英池。

每轮100次更新，batch16，生成batch16；按排序池顺序循环、不shuffle。
AdamW学习率1e-5、weight_decay1e-4、默认betas、clip1；
更新原生可训练参数，原本冻结的扩散常量保持冻结。
不继承原生预训练的优化器状态、AMP、scheduler或额外坐标jitter。

直接调用原生 model.get_loss：
坐标MSE + 100×原子扩散后验KL/NLL + 100×键扩散后验KL/NLL，
并保留原生配置启用的键长项；不改原生实现。
某轮 scores.json 评价的是该轮训练前生成的分子，
该轮 checkpoint.pt 是该轮100次更新后的权重。

SFT得分：
0.5×mean(QED_norm, SA_norm, LE1_norm, LE2_norm)
+ 0.5×mean(D1,D2)。

各项使用capped half Gaussian：
较优方向达到mu后得1；较差侧为 exp(-0.5*((value-mu)/sigma)^2)。
QED：mu0.75/sigma0.1，越高越好；原始SA：mu3/sigma0.5，越低越好。
LE：mu0.5/sigma0.1，越高越好。
LE=1.4×XGBoost预测值/氢原子数，不是重原子数；
指纹使用移除显式氢后的Morgan radius2/2048bit，
分母通过给该重原子分子AddHs后得到的氢数计算，氢数为0明确失败。
SA使用RDKit标准SA算法和指定fpscores表，不使用近似替代公式。

## NFT

只能从本包同源、同靶点、完成8轮的SFT检查点初始化。
当前模型theta训练；old用于生成并按EMA更新；
reference固定为初始SFT8，不是985000步原始预训练。

默认100轮，每轮128次尝试，生成batch16，训练4遍，
micro-batch64，lr1e-5，weight_decay1e-4，clip1。
有效评分少于2个时跳过更新。
保留全部有效样本，并保留至多同等数量的可训练失败图；
空图/非有限终态仍记录失败，但不能训练。

只用有效样本奖励计算：
adv=(R-mean_valid)/(population_std_valid+1e-4)，裁剪至[-1,1]，
r=0.5+0.5adv；保留失败样本r=0。
奖励是两个靶点对接归一化值的算术平均；失败为-1。
QED/SA记录为诊断指标，不参与NFT优化；NFT不加载LE模型。

同一带噪输入和整数时间步输入theta/old/reference，
每个训练epoch重新使用原生q加噪，不采完整生成轨迹或计算策略log概率。
原子、键覆盖全部位置，键按完整图的上三角只计算一次；
不能沿用FlowMol只训练仍mask位置的规则。

正分支 f+=(1-beta)f_old+beta*f_theta；
负分支 f-=(1+beta)f_old-beta*f_theta，beta=1。
两分支均对同一干净图重建，组合 r*L(f+)+(1-r)*L(f-)。
坐标按每个原子sumXYZ(error²)/stopgrad(meanXYZ(abs(error)).clamp_min(1e-5))；
原子、键为原始logits交叉熵。权重分别3/0.4/2，无独立电荷分支。
另加0.001×当前与reference原始输出的均方差之和，
这是输出MSE约束，不是概率KL；不会对预测额外去质心。
micro-batch使用全组原子、半边和特征元素数量作分母，而非简单平均各小批次。

成功外层更新一次后：
eta=min(0.001*successful_updates,0.5)，old=eta*old+(1-eta)*theta。
原生冻结常量和buffers直接复制，不对噪声日程做EMA。
按包含失败-1的总体平均奖励保留Top10，同时保存last/best/final。
日志奖励来自更新前old，保存权重为更新后；元数据明确标注此对齐差异，
Top10训练日志排名不代替独立固定采样评价。

## 采样与分子对应关系

保留原生1000步相邻后验采样和原生重原子数正态分布，
只对不合法的非正原子数重抽并记录；不使用FlowMol显式氢直方图。
新采样器不分配完整1001帧轨迹，不改变旧 model.sample。
终态选择与原生脚本一致：最后一次网络clean预测，而不是最后一次离散抽样状态。

保存原始logits/坐标、mask删除映射和完整上三角训练图。
mask原子被删除；mask边与no-bond映射为无键。
显式键重建后，必须核对元素、原子顺序、键类型/拓扑与坐标一致；
无自动补键。允许原生重建赋予表示中没有独立分支的形式电荷/氢，并记录；
若重建改变了模型表示中的键类型/拓扑，则作为失败负样本，不偷偷用修复分子替换。
不连通或价态失败的有限图仍可用于负分支。
采样中间预测出现非有限值时立即停止，不伪报完成128次尝试。

## 对接

三组沿用FlowMol现有受体、中心和24Å盒子，不用原生RL配置的126Å盒子。
对接越低越好，sigma1：
BRD4/CDK2的mu为-10/-11；另外两组均为-10/-10。
默认16个spawn CPU评分worker，每worker顺序执行双靶点对接，
每次Vina cpu1、exhaustiveness8、num_modes1、timeout300秒、seed2024。
全局依赖/资产缺失在创建run前失败；逐分子评分失败记录错误并赋-1。
仅有schema检查通过不等于这些服务器资产已验证。

## 保存、失败和恢复

任何运行都要求NEW输出目录，包括resume，不覆盖旧run、权重或PBS。
仅支持完整轮次/迭代边界恢复：

    python -B -m mdrl_finetune sft --config SAME_CONFIG.yaml --resume OLD_RUN/checkpoints/last.pt --output NEW_RUN

恢复当前模型、优化器、完整SFT候选池，或NFT的old/reference、
成功更新计数和Top10索引，以及Python/NumPy/Torch/CUDA RNG。
核对stage、配置指纹、源权重SHA256和架构。
不支持半轮精确恢复；失败半轮需从上一安全边界重新运行，旧目录完整保留。
配置须相同；不自动允许修改学习率、靶点、资源配置或延长轮数。
RNG恢复不构成跨硬件/CUDA算子的逐位确定性保证。
评分资产的路径受配置指纹保护，但没有对受体/LE/SA内容计算SHA256；
原路径下替换资产会破坏复现，应保持输入资产不可变。

round_XX/iteration_XX下保存samples.pt（包含失败图和raw）、
scores.json、generated.sdf、metrics.json；
SFT另有retained.pt、retained.json、该轮checkpoint.pt。
run根目录有config.resolved.yaml、status.json、metrics.jsonl、
checkpoints/last.pt与final.pt；NFT另保留Top10和best.pt。
标准输出逐轮输出JSON；异常traceback写failure.json。
检查reward/loss、梯度、clip范数、更新后的模型/优化器/EMA/reference；
异常停止并标FAILED，不发布异常final，last仍为上一安全边界。
失败后不会在内存中回滚部分更新，必须重新进程加载安全边界。

## 本地验证边界

已测试真实CPU优化器、原生扩散kernel、原生get_loss函数体、
正负分支精确公式/梯度、全组micro分母、冻结/EMA、
1000步调用、不补采、图映射、评分公式、检查点/RNG恢复及旧入口回归。
小型CPU预测头、替代分子重建/对接分别明确用于相应测试；
不是完整NodeEdgeNet/KAN模型或真实Vina/GPU验收。
本地缺少torch_geometric、easydict、OpenBabel和XGBoost等完整依赖。
尚未加载实际985000步服务器权重，未进行真实生成、对接、正式训练、
PBS生成/提交、Git提交/推送或服务器操作。

2026-11-12本地最终完整回归：163 passed、2 skipped、55 subtests passed。
其中新增32项测试；两项跳过为当前环境没有可用CPU GradScaler的原有测试。
测试命令为 python -B -m pytest -q -p no:cacheprovider；
原有Git Bash/MSYS测试需要正常本地权限，受限权限下的失败已分别复核。
六份YAML的schema-only入口通过；不等于真实服务器输入预检通过。

SA_Score兼容：优先导入 `rdkit.Contrib.SA_Score.sascorer`，旧版布局回退到
`RDConfig.RDContribDir/SA_Score/sascorer.py`，并兼容旧 loader 对
`fpscores.pkl.gz` 的追加后缀行为。服务器预检仍须确认 RDKit/OpenBabel/Vina、
受体、LE 模型和指定 SA score 表均可用。

本次兼容修复：微调相关40项通过，独立评分兼容回归14项通过。
完整仓库回归为165 passed、2 skipped、55 subtests passed，另6项Git Bash/MSYS
测试在受限权限下失败；提升权限单独复跑该6项全部通过（合计171项通过）。
作者已回传服务器985000步权重CPU严格加载及有限性检查通过；
本修复尚待同步后重新执行三组SFT的`--validate-only`，不代表GPU/对接/训练验收。
