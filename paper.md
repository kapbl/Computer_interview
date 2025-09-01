    
# One-step Noisy Label Mitigation 一步式噪声标签缓解

Hao Li ${}^{1 * }$ 18th.leolee@gmail.com | Jiayang ${\mathrm{{Gu}}}^{1 * }$ jiayang.barrygu@gmail.com  Jingkuan Song ${}^{1 \dagger  }$ jingkuan.song@gmail.com |  An Zhang ${}^{2}$ anzhang@u.nus.edu Lianli ${\mathrm{{Gao}}}^{1}$ lianli.gao@uestc.edu.cn

${}^{1}$ University of Electronic Science and Technology of China  ${}^{2}$ National University of Singapore

# 摘要

<span style="font-size: 18px;">

减轻噪声标签对训练过程的有害影响变得⽇益重要，因为在⼤规模预训练任务中获取完全非噪声或人⼯标注的样本通常不切实际。然⽽，现有的噪声缓解方法在实际应⽤中常因其任务特定设计、模型依赖性和⼤量计算开销⽽受到限制。在本⼯作中，我们利⽤高维正交的特性，在锥空间中识别⼀个鲁棒且有效的边界，⽤于区分非噪声样本和噪声样本。在此基础上，我们提出了⼀步抗噪（OSA）方法，这是⼀种模型无关的噪声标签缓解范式，采⽤估计模型和评分函数，通过⼀次推理评估输入对的噪声水平，过程高效。我们通过实验证明OSA的优越性，突出其增强的训练鲁棒性、改进的任务迁移能⼒、易于部署以及在各种基准、模型和任务中的计算成本降低。

* 锥空间：通过一种特定的数学转换（例如深度学习中的嵌入层），我们将复杂的高维数据（如文本或图像）转换为更低维、更紧凑的向量形式。这些向量在空间中的分布形状像一个​​狭窄的圆锥体​​，并且在这个过程中，数据点之间原始的​​余弦相似度关系​​被保留了下来。
</span>

# 1 引言
<span style="font-size: 18px;">

传统的缓解方法遇到了几个限制其实际适用性的局限性：
- 任务特异性：现有方法[1，3，6]是针对特定任务量身定制的，从而限制了它们在不同任务中的适用性。  
- 模型依赖性：大多数缓解噪声技术[5，7]与特定模型紧密结合，需要对不同模型进行广泛的修改。  
- 计算成本：许多现有方法需要双模型协作[1，4]或多个训练步骤[1]，即，它们每个训练步骤至少需要两个反向梯度传播，有效地增加了计算费用并大大增加了训练负担（请参见图1a）。

为应对这些挑战，我们采⽤外部估算器评估每个样本的噪声水平，确保方法与模型无关。该估算器通过减⼩噪声样本的影响来调整训练损失，推动其权重趋近于零。此外，多模态预训练模型因其强⼤的语义能⼒，在任务迁移方⾯表现出色。
例如，CLIP[8]通过共享嵌入空间统⼀了图像-文本检索和图像分类的范式（⻅图1b）。它将类别标签转化为句子，映射到共享嵌入空间，然后计算与图像表⽰的余弦相似度以进⾏图像分类。
受到此启发，我们利⽤多模态预训练模型作为估计器，并应⽤共享嵌入空间实现任务迁移。在这种情况下，每个样本只需额外进⾏⼀次推理过程，显著减少了计算开销，相比于执⾏额外的反向传播。
##### ⚡然⽽，这⼀范式引入了⼀个新挑战：如何仅凭估计器⽣成的余弦相似度分数准确识别噪声？
理想的解决方案是找到⼀个决策边界，将非噪声样本与噪声区分开，并能准确处理边界附近的重叠样本。现有方法 通常试图在损失空间中构建此边界，该空间是⼀个各向同性、分布均匀的空间，这导致噪声样本与非噪声样本之间的间隙很窄。此外，通过整合多模型预测对重叠部分的粗略处理，常常导致决策边界不稳定。相比之下，预训练模型的共享嵌入空间是⼀个高维、各向异性的空间，分布不均衡。因此，是否可以利⽤不平衡各向异性空间的特性，帮助识别更精确、更稳健的决策边界，很值得考虑。
在本研究中，我们深入分析了⽤作估计器的预训练模型的决策边界，以准确区分非噪声样本和噪声样本。我们⾸先研究了在两个噪声比例为50%的数据集上，利⽤多模态预训练模型CLIP [8]和ALIGN [11]计算的非噪声样本和噪声样本的余弦相似度分布：MS-COCO [12]和SDM，如图1c-1f所⽰。
SDM是⼀个由稳定扩散模型（SDM）[13]在⼀些不常⻅⻛格下⽣成的图像组成的数据集（⻅图4中的插图）。其设计旨在探索预训练模型在区分训练中少⻅领域时的表现。图1c-1f中有两个有趣的观察：
(1)相同模型在不同数据集上的非噪声和噪声分布具有相似的交点，表明存在⼀个⾃然且稳定的边界，⽤于区分非噪声样本和噪声样本。
(2)即使在不熟悉的领域数据集上，非噪声分布与噪声分布的重叠也很⼩，表明该边界在区分非噪声与噪声样本方⾯具有很强的潜⼒。
基于这两个观察，我们进⾏了深入的研究并做出以下贡献：

- 我们确定了交点的起源，将其归因于锥效应引起的正交边界的偏移。此外，我们提供了⼀个理论框架，证明并阐述了该边界在区分噪声样本和非噪声样本方⾯的稳定性和精确性。
- 我们基于对预训练过程的分析，详细说明了预训练模型在⼀般噪声识别中的可靠性，即使在不熟悉的领域也是如此。
- 在此基础上，我们提出了⼀步抗噪（OSA）方法，这是⼀种通⽤的模型无关噪声识别范式，仅需⼀步推理。具体⽽言，我们利⽤预训练模型作为估计器，以保持共享的嵌入空间。然后，基于高维正交性质设计的评分函数，⽤于通过直接为每个样本的损失分配学习权重（根据其余弦相似度）来准确处理重叠问题。
- 我们在多种具有挑战性的基准、模型和任务上进⾏了全⾯实验，验证了我们方法的有效性、泛化能⼒和效率。

</span>
<div align="center">

### 图1：(a) 当前的抗噪声范式通过多次反向传播显著增加了训练开销。(b) CLIP通过共享空间统一了图像-文本匹配和图像分类的框架。(c-f) 在50%噪声条件下噪声数据和干净数据的余弦相似度分布。
</div>
<div align="center">
<img src="image\image1.png"/>
</div>


# 2 边界原理分析
<span style="font-size: 18px;">
在图1c-1f中，我们观察到预训练模型在区分非噪声样本和噪声样本时⾃然形成的边界。在本节中，我们从高维⻆度解释边界形成的原理，以及其在⼀般噪声抑制中的鲁棒性。
</span>

### 2.1 假设：交互边界从正交边界偏移
<span style="font-size: 18px;">
我们⾸先阐述正负两侧由正交边界所保持的差距范围。然后，提出假设：图1中的交点边界是在锥空间中偏移的正交边界的推理依据。正交边界在很⼤程度上分隔了正负两侧。高维正交性是⼀种普遍现象，由维度灾难引起，随机选择的向量之间的夹⻆通常接近90度，表明余弦相似度趋向于零。例如，在1024维空间中，两个随机向量的余弦相似度在[−0.1，0.1]范围内的概率约为99.86%[6]。在这种情况下，形成⼀个⾃然的余弦相似度为零的边界，有效地将正负两侧分隔开来，差距巨⼤。
</span>

###  锥体效应可能引起正交边界的偏移。 
<span style="font-size: 18px;">
近期文献[14-16]表明，锥效应是深度神经网络中的普遍现象，学习到的嵌入子空间形成⼀个狭窄的锥体，正交边界出现正向偏移。基于此，假设图1中的交互边界是偏移的正交边界。为了验证这⼀点，我们模拟在高维空间中随机选择向量的过程，并随机⽣成数千对映射到共享嵌入空间的向量。我们发现这些随机向量对的相似度趋于⼀个固定值，低方差的余弦相似度⼏乎位于非噪声和噪声分布的中间（⻅表1）。⼀个有趣的现象是，如果我们比较图1c-1f中的均值与交点，我们会发现它们⼏乎完全相同。这表明交互边界很可能是锥空间中偏移的正交边界。
</span>

<div align="center">

### 表1：随机生成对之间余弦相似度的均值和方差。
</div>
<div align="center">
<img src="table\table1.png"/>
</div>

### 2.2 交互边界起源的理论验证

<span style="font-size: 18px;">

在此，我们理论上探讨了交互边界的起源是否为偏移的正交边界。我们⾸先证明
（i）对比学习在正交边界的两侧区分非噪声样本和噪声样本。
（ii）成对样本的余弦相似度的相对关系在传入狭窄锥空间后保持不变。
基于（i）和（ii），我们可以确认非噪声和噪声分布中心的交点边界即为偏移的正交边界。对比学习增强了非噪声样本和噪声样本的分离能⼒。由于在初始空间中缺乏语义感知能力，使用了随机向量。在对比训练过程中，给定 $N$ 个样本对 ${\left\{  \left( {x}_{i},{y}_{i}\right) \right\}  }_{i = 1}^{N}$ ，嵌入空间通过交叉熵损失（式1）进行优化。

$$
{\mathcal{L}}_{ce} = \frac{1}{N}\mathop{\sum }\limits_{{i = 1}}^{N}\log \frac{\exp \left( {m}_{ii}\right) }{\mathop{\sum }\limits_{{j = 1}}^{N}\exp \left( {m}_{ij}\right) }, \tag{1}
$$

其中 $M \in  {\mathbb{R}}^{N \times  N}$ 代表训练过程中余弦相似性矩阵。 每个元素 ${m}_{ij} \in  M$ 表示 ${x}_{i}$ 和 ${y}_{j}$ 之间的余弦相似性。 对角元素 ${m}_{ii}$ 表示正向相关对的余弦相似性，而非对角线元素 ${m}_{ij}$ 表示负相关对的余弦相似性。为了最大程度地减少训练期间的 ${\mathcal{L}}_{ce}$，发生了两个子过程：将矩阵的对角元素(即，干净对)优化到正交边界的正侧，而将非对角元素(相当于噪声对)优化到负侧。因此，这两种类型的样本的分布在正交边界的相反侧。
</div>
<div align="center">
<img src="image\ii1.png" width="50%"/>
</div>
</span>

#### 相对关系在传输过程中没有变化。 
<span style="font-size: 18px;">
我们研究边界如何从整个空间转移到神经网络中的狭窄锥体。 以下定理表明，余弦的相似性将按比例缩放到目标窄锥，同时仍具有类似于正交边界的属性的边界。 换句话说，余弦相似性小于原始空间中的正交边界的矢量仍然小于狭窄的圆锥空间中移动的边界，而较大的边界则更大。

#### Theorem 1 (边界比例移动). 
设 ${\mathbb{R}}^{{d}_{in}}$ 为神经网络传输前的原始空间。假设 $u,v \in  {\mathbb{R}}^{{d}_{in}}$ 是任意两个随机向量，满足 $\cos \left( {u,v}\right)  \approx  0$。${u}_{c},{v}_{c} \in$ ${\mathbb{R}}^{{d}_{in}}$ 是一对干净向量，满足 $\cos \left( {{u}_{c},{v}_{c}}\right)  > 0$，而 ${u}_{n},{v}_{n} \in  {\mathbb{R}}^{{d}_{in}}$ 是一对噪声向量，满足 $\cos \left( {{u}_{n},{v}_{n}}\right)  < 0$。给定一个具有 $t$ 层的神经网络 $F\left( x\right)  = {f}_{t}\left( {{f}_{t - 1}\left( {\ldots {f}_{2}\left( {{f}_{1}\left( x\right) }\right) }\right) }\right)  \in  {\mathbb{R}}^{{d}_{\text{out }}}$。${f}_{i}\left( x\right)  = {\sigma }_{i}\left( {{\mathbf{W}}_{i}x + {\mathbf{b}}_{i}}\right)$ 表示第 ${i}$ 层，其中 $\sigma \left( \cdot \right)$ 表示激活函数。${\mathbf{W}}_{i} \in  {\mathbb{R}}^{{d}_{\text{out }}^{i} \times  {d}_{\text{in }}^{i}}$ 是一个随机权重矩阵，其中每个元素 ${\mathbf{W}}_{i}^{k,l} \sim  \mathcal{N}\left( {0,1/{d}_{\text{out }}^{i}}\right)$，对于 $k \in  \left\lbrack  {d}_{\text{out }}^{i}\right\rbrack  ,l \in  \left\lbrack  {d}_{\text{in }}^{i}\right\rbrack$，${\mathbf{b}}_{i} \in  {\mathbb{R}}^{{d}_{\text{out }}^{i}}$ 是一个随机偏置向量，满足 ${\mathbf{b}}_{i}^{k} \sim  \mathcal{N}\left( {0,1/{d}_{\text{out }}^{i}}\right)$，对于 $k \in  \left\lbrack  {d}_{\text{out }}^{i}\right\rbrack$。那么，总是存在一个边界 $\beta$，满足：

$$
\cos \left( {F\left( {u}_{n}\right) ,F\left( {v}_{n}\right) }\right)  < \cos \left( {F\left( u\right) ,F\left( v\right) }\right)  \approx  \beta  < \cos \left( {F\left( {u}_{c}\right) ,F\left( {v}_{c}\right) }\right) . \tag{2}
$$

定理1表明，原始整个空间中成对样本的相对关系在传输到训练模型的狭窄锥空间后不会改变，并且总是存在一个集中在大多数随机向量上的边界 $\beta$。
</span>

### 2.3 鲁棒性和适用性的定性分析
<span style="font-size: 18px;">

接下来，我们进行定性分析来探索（i）边界在区分干净样本和噪声样本方面的鲁棒性和通用性，以及（ii）如何利用边界的特性来实现更合理和精确的重叠处理。即使在不熟悉的领域中，边界的鲁棒性如何？虽然边界区分干净样本和噪声样本的能力已得到证明，但其鲁棒性和通用性仍需进一步探索。对于实际的预训练，它必须在不熟悉的领域数据集上也保持准确性和鲁棒性。由于预训练模型的能力难以量化，我们从预训练模型推理的角度进行定性分析。在数百万样本上预训练的模型已经具备了一定的语义理解能力。给定来自未见领域的正样本对，由于预训练期间的对比学习过程，它仍然很有可能向边界的正侧移动，而负样本对则倾向于负侧。虽然余弦相似度差异可能很小，但正如我们在第2.1节中所示，从高维正交性的角度来看，边界构建了一个显著的间隙。
如何通过不平衡概率处理重叠？由于正交边界的特性，当余弦相似度从正侧减少并接近零时，正样本的概率急剧下降。因此，我们可以设计一个评分函数来标注样本的清洁度。该函数应满足两个要求：对于余弦相似度小于或等于零的样本（几乎肯定是噪声），函数应为其分配零权重。对于余弦相似度大于零的样本，当余弦相似度远离零时，函数梯度应快速增加。
</span>

## 3 方法
<span style="font-size: 18px;">
在本节中，我们提出了一步抗噪声（OSA）范式，其工作流程如图2所示。我们首先在第3.1节中定义了图像-文本匹配、图像分类和图像检索任务的基于成对的噪声缓解任务。随后，在第3.2节中阐明了OSA的详细描述。
</span> 

### 3.1 任务定义
<span style="font-size: 18px;">

设 $\mathcal{D} = {\left\{  \left( {x}_{i},{y}_{i},{c}_{i}\right) \right\}  }_{i = 1}^{N}$ 表示一个成对数据集，其中 $\left( {{x}_{i},{y}_{i}}\right)$ 代表数据集中第 $i$ 个样本对，${c}_{i}$ 表示该样本对的噪声标签。具体地，当 ${c}_{i} = 0$ 时，$\left( {{x}_{i},{y}_{i}}\right)$ 形成正确的（配对的）匹配，而 ${c}_{i} = 1$ 表示错误的（非配对的）匹配。对比学习中噪声缓解的目标是构建一个共享嵌入空间，当 ${c}_{i} = 1$ 时使 ${x}_{i}$ 和 ${y}_{i}$ 更加接近。在不同的任务中，${x}_{i}$ 和 ${y}_{i}$ 是不同的数据类型。例如，在图像-文本检索任务中，${x}_{i}$ 和 ${y}_{i}$ 分别表示图像和文本。在图像分类任务中，${x}_{i}$ 和 ${y}_{i}$ 分别表示图像和类别。在图像检索任务中，${x}_{i}$ 和 ${y}_{i}$ 分别表示图像和相关图像。成对样本(x,y)可以通过相应的编码器 ${\phi }_{x}\left( \cdot \right)$ 和 ${\phi }_{y}\left( \cdot \right)$ 编码到共享嵌入空间中。然后，通过公式3计算余弦相似度 $s\left( {x,y}\right)$ 作为(x,y)的语义相关性来指导训练。

$$
s\left( {x,y}\right)  = \frac{{\phi }_{x}\left( x\right) }{\begin{Vmatrix}{\phi }_{x}\left( x\right) \end{Vmatrix}} \cdot  \frac{{\phi }_{y}\left( y\right) }{\begin{Vmatrix}{\phi }_{y}\left( y\right) \end{Vmatrix}}. \tag{3}
$$
</span>

### 3.2 一步式抗噪声
<span style="font-size: 18px;">

我们的噪声缓解方法OSA的工作流程如图2所示。首先，我们利用估计器模型将输入样本对编码到共享嵌入空间中，并继续计算配对嵌入之间的余弦相似度。然后，通过基于正交特性设计的评分函数（第2.3节），将余弦相似度转换为清洁度分数 ${w}_{i},\left( {0 \leq  {w}_{i} \leq  1}\right)$。该分数量化了样本的清洁程度，${w}_{i}$ 越小，样本越嘈杂。在目标模型训练阶段，该清洁度分数用作权重，直接与相应样本的损失相乘，以促进选择性学习。这种噪声缓解过程仅依赖于估计器模型，通过简单地在损失函数中添加额外系数，就能够轻松适应各种目标模型的训练，确保了模型无关性。因此，我们噪声缓解方法的关键围绕估计器模型和噪声分数评估展开。
</span>

#### 3.2.1 估计器模型
<span style="font-size: 18px;">

估计器模型选择。在我们的方法中，估计器模型必须满足两个关键要求：
1）有效地将输入对映射到统一的嵌入空间；
2）具备基本的语义理解能力。
为了满足这些要求，我们采用CLIP [8]作为我们的估计器模型，这是一个常用的多模态预训练模型。它配备了文本编码器 ${\phi }_{t}\left( \cdot \right)$ 和图像编码器 ${\phi }_{v}\left( \cdot \right)$，使其能够高效地执行基本的零样本任务。领域适应（可选）。虽然我们在第2.3节中对零样本预训练模型在域外数据上的鲁棒性进行了定性分析，并在图1中展示了边缘情况下的强鲁棒性，但考虑到现实场景中的领域多样性，我们提供了一种可选的领域适应（DA）方法来增强估计器模型在遇到边缘领域时的适应性。遵循NPC [2]的做法，我们首先采用高斯混合模型（GMM）结合严格的选择阈值来确保所选样本的绝对清洁性。

<div align="center">

#### 图2：OSA的工作流程。在抗噪声过程中，有两个阶段：评分阶段和训练阶段。在评分阶段，一个样本对通过估计器映射到共享的嵌入空间。然后余弦相似度通过评分函数转换为权重$w$。在训练阶段，权重$w$直接与损失相乘来指导优化。
</div>
<div align="center">
<img src="image\image2.png"/>
</div>

我们随后实现了一个包含少量步骤的预热阶段，允许估计器模型更好地理解目标领域的语义。值得注意的是，这个技巧对我们的方法来说只是可选的。通过多次实验，我们发现即使没有领域适应，零样本CLIP模型在各种场景下都表现得异常出色。
</span>

#### 3.2.2 噪声评分评估
<span style="font-size: 18px;">

#### 空间去偏。
锥效应现象已被证明是深度神经网络的一般现象，通常导致狭窄的嵌入空间，使空间中心偏移到狭窄的锥中心[14]。具体来说，当成对的随机生成输入通过模型编码器映射到共享嵌入空间时，结果向量表现出偏离零的平均余弦相似度，并趋向于另一个固定角度。为了抵消这种偏移并减轻其对估计器通过高维正交性准确识别噪声能力的影响，开发了一种随机采样方法。我们首先构造$K$个随机样本对 $\mathcal{R} = \left\{  {\left( {{x}_{j},{y}_{j}}\right)  \mid  j = 1,2,\ldots ,K}\right\}$，并通过估计器的编码器处理它们以生成一组向量。然后通过以下方式计算这些向量之间的平均余弦相似度作为空间偏移$\beta$：

$$
\beta  = \frac{\mathop{\sum }\limits_{{j = 1}}^{K}s\left( {{x}_{j},{y}_{j}}\right) }{K}. \tag{4}
$$

#### 评分函数。
在空间去偏之后，我们采用评分函数 $w\left( \cdot \right)$ 来评估输入对(x,y)的纯度。在第2.3节中，我们详细阐述了如何基于正交边界特性处理重叠。对于使用对比学习在数百万样本上训练的估计器模型，非噪声对（对角元素）被优化到正侧，而噪声对（非对角元素）被优化到负侧。给定不熟悉的对，模型也倾向于将非噪声对映射到正侧，将噪声对映射到负侧。尽管非噪声对和噪声对之间的相似度差异可能很小，但高维正交性确保了它们之间的实质性差距。在这种情况下，由估计器计算的负余弦相似度  $s\left( {x,y}\right)$ 表明该对几乎肯定是噪声，应该被分配零分。对于 $s\left( {x,y}\right)$ 大于零的样本，当余弦相似度从正侧接近零时，样本为正的概率急剧下降。因此，当余弦相似度远离零时，函数梯度应该快速增加。为了系统地对噪声进行评分，我们设计评分函数如下：

$$
w\left( {x,y,\beta }\right)  = \left\{  \begin{array}{ll} 0 & ,s\left( {x,y}\right)  - \beta \\   - {\left( s\left( x,y\right)  - \beta \right) }^{2}\left( {s\left( {x,y}\right)  - \beta  - 1}\right) & ,\text{ otherwise } \end{array}\right.  \tag{5}
$$

#### 重新加权训练。
在评分之后，目标模型可以通过重新加权损失来选择性地从样本中学习。权重较小的噪声样本对模型更新的影响将减少，并将被有效缓解。对于样本(x,y)，设${\mathcal{L}}_{x,y}$表示其损失，重新计算的损失

${\mathcal{L}}_{re}$ 被定义为:

$$
{\mathcal{L}}_{re} = w\left( {x,y,\beta }\right)  \times  {\mathcal{L}}_{x,y}. \tag{6}
$$

</span>

## 4 实验

在本节中，我们在多个带有标签噪声的数据集上进行实验，展示了我们方法的有效性。首先，我们描述数据集、指标和实现细节。然后，我们报告在几个下游任务上的结果。最后，我们进行消融研究，展示我们方法的每个部分如何贡献以及这些部分如何相互作用。

### 4.1 评估设置

在本节中，我们简要介绍实验中使用的数据集和评估指标。有关更多数据集和实现细节。

📦数据集。
我们在三个带有噪声标签的下游任务上评估我们的方法，包括一个多模态任务和两个视觉任务。
* 对于跨模态匹配任务，我们在MSCOCO [12]和Flickr30K [17]数据集上进行实验。遵循NPC [2]，我们进一步在真实世界噪声数据集 $\underline{\mathrm{{CC}}}{120}\mathrm{\;K}$ 上进行评估。
* 对于图像分类任务，实验在WebFG-496 [3]的三个子集——Aircraft、Bird和Car上进行。
* 对于图像检索任务，我们在PRISM [5]设置下的CARS98N数据集上进行实验。

📏评估指标。
* 对于图像-文本匹配任务，使用前K个检索结果的召回值(R@K)。
* 对于分类任务，准确率作为评估指标。
* 对于图像检索任务，我们使用Precision@1和mAP@R进行评估。

### 4.2 与最先进方法的比较

<!-- Media -->
<div align="center">

#### 表2：在噪声MS-COCO上的比较。
</div>

<div align="center">
<img src="table\table2.png"/>
</div>  
<!-- Media -->
<span style="font-size: 18px;">

MSCOCO上的结果。为了公平地展示我们方法的有效性，我们将OSA与各种使用相同ViT-B/32 CLIP作为骨干网络的鲁棒学习图像-文本匹配方法进行比较，包括VSE $\infty$ [18]、PCME++ [19]、PAU [20]、NPC [2]。此外，我们分别在CLIP [8]和ALIGN [11]上应用OSA。表2中的结果显示，OSA在所有指标上都以巨大优势超越了所有先前的方法。在更具挑战性的MS-COCO 5K数据集上，当噪声比例为 ${50}\%$ 时，OSA在图像到文本(i2t)和文本到图像(t2i)匹配的R@1指标上分别超越了最先进方法NPC ${8.6}\%$ 和 ${7.0}\%$ 。另一个现象是，当噪声比例从 $0\%$ 增加到 ${50}\%$ 时，所有其他方法都遇到了严重的性能下降，NPC在四个R@1指标上的平均下降幅度为 ${5.05}\%$。相比之下，OSA仅表现出1.275%的轻微下降，展现了OSA在抗噪声任务中的准确性和鲁棒性。

</span>

<div align="center">

#### 表3：在噪声Flickr30K上的比较。
</div>

<div align="center">
<img src="table\table3.png"/>
</div>  
<!-- Media -->
<span style="font-size: 18px;">

Flickr30K上的结果。为了进一步展示OSA的泛化能力，我们在Flickr30K数据集上进行评估，并与几种抗噪声方法进行比较，包括NCR [1]、DECL [9]、BiCro [7]和NPC [2]。结果在表3中呈现。显然，OSA在R@1指标上始终优于所有模型。值得注意的是，与基线CLIP相比，在 ${60}\%$ 噪声比例下使用OSA训练，$\mathrm{{i2t}}$ 的R@1指标提升了 ${20.9}\%$ ，$\mathrm{{t2i}}$ 的R@1指标提升了 ${22.3}\%$ ，进一步表明了OSA在噪声缓解方面的有效性。此外，OSA在Flickr30K数据集上表现出与在MSCOCO上观察到的类似噪声鲁棒性，从0%噪声到60%噪声范围内，$\mathrm{{i2t}}$的R@1仅下降1.4%，$\mathrm{{t2i}}$的R@1仅下降1.2%，而所有其他抗噪声方法几乎无法抵抗高比例噪声的损害。所有这些结果都证明了OSA在抗噪声任务中的有效性和鲁棒性。

CC120K上的结果。为了进一步验证OSA在真实场景中的可靠性，我们在大规模真实世界噪声数据集CC120K上进行评估，噪声比例为3%-20%。表4中显示的结果表明，即使在更大规模的真实世界领域中，OSA也优于当前最先进的方法NPC。这证明了OSA即使在实际训练场景中的可行性和通用性。

</span>
<!-- Media -->

<div align="center">

#### 表4：在真实CC120K上的比较。
</div>

<div align="center">
<img src="table\table4.png"/>
</div>  

<div align="center">


<span style="font-size: 18px;">

#### 表5：其他图像任务的结果。
</div>
<div align="center">
<img src="table\table5.png"/>
</div>  
其他下游任务的结果。为了验证OSA在不同任务中的可迁移性，我们在两个额外任务上对其进行评估：图像分类和图像检索。结果在表5中呈现。两个任务的基线方法都利用了对比学习。在图像分类任务中，OSA在Aircraft、Bird和Car子集上分别超越基线7.74%、8.21%和4.28%。在图像检索任务中，OSA在精确度上提升了6.76%，在mAP上提升了6.83%。这些改进证明了OSA强大的任务可迁移性和通用性。

</span>

<div align="center">

#### 表6：在噪声MSCOCO上不同架构的目标模型的结果。
</div>
<div align="center">
<img src="table\table6.png"/>
</div>  

### 4.3 目标模型无关性分析

<span style="font-size: 18px;">
OSA是一个架构无关的范式，可以轻松适应各种模型。为了验证其模型无关性，我们在不同架构的模型上对其进行评估。随后，我们将其应用于其他抗噪声模型，以证明其在噪声缓解方面的泛化能力。架构无关性分析。OSA在视觉变换器(ViT)上的有效性已在第4.2节中得到证明。我们进一步探索OSA在其他架构的目标模型上的通用性。具体而言，我们在VSE++ [21]模型上部署OSA，该模型采用两种不同的架构类型：ResNet-152 [22]和VGG-19 [23]。这两种架构已显示出对噪声的显著敏感性和脆弱性[1]。在此实验中，所有估计器模型都采用零样本CLIP，我们使用原始VSE++作为基线。表6中的结果表明，基线方法在噪声环境中出现了显著的性能下降，而采用OSA后实现了稳定的性能。在这两种对噪声脆弱的架构上的稳定性能充分证明了OSA具有架构无关性。

</span>

<!-- Media -->

<div align="center">

#### 表7：在MSCOCO 1K上采用OSA的其他方法的结果。
</div>
<div align="center">
<img src="table\table7.png"/>
</div>  
<!-- Media -->
<span style="font-size: 18px;">

对其他抗噪声模型的适应性。理论上，OSA可以适应任何目标模型，提供噪声抵抗能力。然而，OSA能否进一步增强专门为噪声缓解设计的模型的鲁棒性？为了研究这一点，我们将OSA应用于当前最先进的模型NPC [2]。如表7所示，即使对于噪声缓解模型，OSA也能持续改善训练鲁棒性。这一发现进一步证明了OSA在不同模型类型中的广泛适应性。

</span>

### 4.4 估计器模型分析

<span style="font-size: 18px;">
估计器模型是OSA抗噪声能力的基础。在本节中，我们探讨了不同估计器模型对噪声缓解的影响，并检验了领域适应在噪声缓解中的作用。我们研究了四种类型的估计器："None"指直接训练CLIP而不使用OSA。"CLIP (w/o DA)"和"ALIGN (w/o DA)"分别表示使用CLIP和ALIGN作为估计器而不进行领域适应，即零样本CLIP和ALIGN。"CLIP (w DA)"表示进行领域适应的CLIP。目标模型均为CLIP。我们可以观察到，CLIP和ALIGN作为估计器都显著增强了目标模型在噪声学习中的性能稳定性，表明估计器的选择非常灵活。CLIP和ALIGN作为估计器都表现出卓越的性能。另一个现象是，零样本CLIP模型显示出与领域适应CLIP相当的性能，在较低噪声比率下甚至表现更好。这表明零样本CLIP作为估计器在噪声缓解方面已经表现得异常出色。领域适应是不必要的。这进一步增强了OSA的部署便利性。
</span>

<!-- Media -->

<div align="center">

#### 表8：在噪声MS-COCO上估计器类型的消融研究。
</div>
<div align="center">
<img src="table\table8.png"/>
</div>  
<!-- Media -->

### 4.5 噪声评估准确性
<span style="font-size: 18px;">
噪声检测准确性分析。为了了解OSA在识别噪声方面的准确性，我们在噪声MSCOCO上评估了无领域适应的CLIP（w/o DA）和有领域适应的CLIP（w DA）的准确率和召回率。我们使用零作为阈值来粗略地将配对分为噪声集和干净集。具体来说，我们将小于或等于0的分数分类为噪声，将大于0的分数分类为干净。准确率是指正确分类到干净集中的干净配对的比例，而召回率是指正确分类到噪声集中的噪声配对的比例。表10中呈现的结果表明了OSA强大的噪声识别能力。CLIP（w/o DA）的卓越性能充分证明了OSA的通用性。另一个值得注意的现象是，所有召回率分数都趋向于100，表明OSA在噪声检测方面实现了更高的准确性。这表明OSA几乎可以完全消除噪声对训练的影响。
</span>

<!-- Media -->

<div align="center">

#### 表9：OSA和NPC的平均噪声排名比较。
</div>
<div align="center">
<img src="table\table9.png"/>
</div>  
<span style="font-size: 18px;">
噪声重新加权准确性比较。一些抗噪声方法，如NPC，也采用损失重新加权进行优化。为了评估我们的方法是否比这些方法为噪声分配相对较小的权重，我们首先分析NPC和OSA生成的权重。由于不同方法之间权重尺度的差异，直接比较是不公平的。因此，为了统一尺度，我们采用基于排名的方法，按降序对权重进行排序并计算平均噪声排名。该指标评估是否相对于干净样本，较小的权重被一致地分配给噪声样本。我们的实验使用从MSCOCO数据集中随机选择的2,000个样本，在两种噪声条件下：20% 噪声（370个噪声样本）和50%噪声（953个噪声样本）。理论最优平均噪声排名，即所有噪声权重都排在最后，分别为1815.5和1524.0。与NPC相比，OSA实现了更高的平均噪声排名，证明了在重新加权方面具有更高的准确性。此外，OSA的排名几乎是最优的（20%噪声：OSA为1809.1对比最优1815.5；50%噪声：OSA为1520.7对比最优1524.0）。这种近乎完美的对齐表明OSA有效地将几乎所有噪声样本放在干净样本之后。
</span>

<div align="center">

#### 表10：ACC和噪声检测的recall。
</div>
<div align="center">
<img src="table\table10.png"/>
</div>  
<div align="center">

#### 表11：框架比较。
</div>
<div align="center">
<img src="table\table11.png"/>
</div>  

### 4.6 Computational Cost Analysis

<span style="font-size: 18px;">
预训练中的成本。为了评估OSA在真实世界预训练场景中的实用性，我们估算了处理10亿个数据点的额外计算成本。使用NVIDIA RTX 3090，推理批次大小为4096，利用约24 GB的GPU内存，处理包含566,435对的MS-COCO数据集大约需要153秒。按照这个推理速度，处理10亿个数据点在单个RTX 3090上大约需要75小时。在大规模预训练的背景下，这个成本是微不足道的，特别是在利用多个GPU进行并行推理时。时间成本比较。为了进一步检验OSA与其他抗噪声技术相比的计算效率，我们评估了与两种代表性方法的训练时间：CLIP和NPC。CLIP作为基线，直接训练而不使用任何额外技术。NPC是当前最先进的方法，也使用CLIP作为骨干网络，但通过估计每个样本的负面影响来应用抗噪声技术，需要双重反向传播。表11中呈现的训练时间比较显示，我们的方法与直接训练相比只引入了最小的训练时间增加，仅需要NPC所需额外时间的十分之一。这突出了OSA的效率，使其非常适合大规模鲁棒训练。
</span>

## 5 结论

<span style="font-size: 18px;">

#### 更广泛的影响。
在这项工作中，我们研究了在实际大规模训练中抗噪声的可能性。我们引入了一种新颖的模型无关抗噪声范式，具有任务可转移性、模型适应性和低计算开销等优势。通过利用高维空间的特性，我们找到了一个鲁棒且有效的边界来区分噪声样本和干净样本。通过严格的理论分析和全面的实验，我们验证了OSA在一般噪声缓解方面的有效性和鲁棒性。尽管我们的主要目标是适应实际的大规模训练，但OSA在标准抗噪声设置中也达到了最先进的性能。据我们所知，这是第一个在实际大规模训练场景中探索抗噪声的工作，也是第一个提出通用抗噪声方法的工作。

#### 局限性和未来工作。
由于预训练的巨大计算成本限制，我们很难在真实的预训练过程中进行评估。相反，我们尽可能地模拟大规模预训练过程，例如在真实世界的噪声数据集CC120K上进行评估，该数据集与主流预训练数据集（如CC4M和CC12M）共享相似的领域。在真实预训练场景中探索OSA的广泛领域适应性将是未来工作的一个有价值的方向。 References
</span>

## 6 文献
[1] Zhenyu Huang, Guocheng Niu, Xiao Liu, Wenbiao Ding, Xinyan Xiao, Hua Wu, and Xi Peng. Learning with noisy correspondence for cross-modal matching. In NeurIPS, pages 29406-29419, 2021. 1, 2, 8, 9, 15, 16

[2] Xu Zhang, Hao Li, and Mang Ye. Negative pre-aware for noisy cross-modal matching. In AAAI, pages ${7341} - {7349},{2024.1},2,5,7,8,9,{15},{16}$

[3] Zeren Sun, Yazhou Yao, Xiu-Shen Wei, Yongshun Zhang, Fumin Shen, Jianxin Wu, Jian Zhang, and Heng Tao Shen. Webly supervised fine-grained recognition: Benchmark datasets and an approach. In ICCV, pages 10582-10591. IEEE, 2021. 1, 7, 15

[4] Xingrui Yu, Bo Han, Jiangchao Yao, Gang Niu, Ivor W. Tsang, and Masashi Sugiyama. How does disagreement help generalization against label corruption? In ICML, pages 7164-7173, 2019. 1

[5] Chang Liu, Han Yu, Boyang Li, Zhiqi Shen, Zhanning Gao, Peiran Ren, Xuansong Xie, Lizhen Cui, and Chunyan Miao. Noise-resistant deep metric learning with ranking-based instance selection. In CVPR, pages 6811-6820, 2021. 1, 7, 15, 16

[6] Sarah Ibrahimi, Arnaud Sors, Rafael Sampaio de Rezende, and Stéphane Clinchant. Learning with label noise for image retrieval by selecting interactions. In WACV, pages 468-477, 2022. 1

[7] Shuo Yang, Zhaopan Xu, Kai Wang, Yang You, Hongxun Yao, Tongliang Liu, and Min Xu. Bicro: Noisy correspondence rectification for multi-modality data via bi-directional cross-modal similarity consistency. In CVPR, pages 19883-19892, 2023. 1, 8, 15, 16

[8] Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, Gretchen Krueger, and Ilya Sutskever. Learning transferable visual models from natural language supervision. In ICML, volume 139, pages 8748-8763, 2021. 2, 5, 8

[9] Yang Qin, Dezhong Peng, Xi Peng, Xu Wang, and Peng Hu. Deep evidential learning with noisy correspondence for cross-modal retrieval. In ${ACMMM}$ ,pages 4948-4956,2022. 2,8,15

[10] Junnan Li, Richard Socher, and Steven C. H. Hoi. Dividemix: Learning with noisy labels as semi-supervised learning. In ${ICLR},{2020.2},{16}$

[11] Chao Jia, Yinfei Yang, Ye Xia, Yi-Ting Chen, Zarana Parekh, Hieu Pham, Quoc V. Le, Yun-Hsuan Sung, Zhen Li, and Tom Duerig. Scaling up visual and vision-language representation learning with noisy text supervision. In ICML, volume 139, pages 4904-4916, 2021. 2, 8

[12] Tsung-Yi Lin, Michael Maire, Serge J. Belongie, James Hays, Pietro Perona, Deva Ramanan, Piotr Dollár, and C. Lawrence Zitnick. Microsoft COCO: common objects in context. In ECCV, volume 8693, pages 740-755, 2014. 2, 7

[13] Robin Rombach, Andreas Blattmann, Dominik Lorenz, Patrick Esser, and Björn Ommer. High-resolution image synthesis with latent diffusion models. In CVPR, pages 10674-10685, 2022. 2

[14] Weixin Liang, Yuhui Zhang, Yongchan Kwon, Serena Yeung, and James Y. Zou. Mind the gap: Understanding the modality gap in multi-modal contrastive representation learning. In NeurIPS, 2022.3, 6, 17

[15] Simion-Vlad Bogolin, Ioana Croitoru, Hailin Jin, Yang Liu, and Samuel Albanie. Cross modal retrieval with querybank normalisation. In CVPR, pages 5184-5195, 2022.

[16] Kawin Ethayarajh. How contextual are contextualized word representations? comparing the geometry of bert, elmo, and GPT-2 embeddings. In EMNLP, pages 55-65, 2019. 3

[17] Peter Young, Alice Lai, Micah Hodosh, and Julia Hockenmaier. From image descriptions to visual denotations: New similarity metrics for semantic inference over event descriptions. Trans. Assoc. Comput. Linguistics, 2:67-78, 2014. 7

[18] Jiacheng Chen, Hexiang Hu, Hao Wu, Yuning Jiang, and Changhu Wang. Learning the best pooling strategy for visual semantic embedding. In CVPR, pages 15789-15798, 2021. 8

[19] Sanghyuk Chun. Improved probabilistic image-text representations. arXiv preprint arXiv:2305.18171, 2023. 8

[20] Hao Li, Jingkuan Song, Lianli Gao, Xiaosu Zhu, and Hengtao Shen. Prototype-based aleatoric uncertainty quantification for cross-modal retrieval. In NeurIPS, 2023. 8

[21] Fartash Faghri, David J. Fleet, Jamie Ryan Kiros, and Sanja Fidler. VSE++: improving visual-semantic embeddings with hard negatives. In ${BMCV}$ ,page 12,2018. 9

[22] Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun. Deep residual learning for image recognition. In CVPR, pages 770-778, 2016. 9

[23] Karen Simonyan and Andrew Zisserman. Very deep convolutional networks for large-scale image recognition. arXiv preprint arXiv:1409.1556, 2014. 9

[24] Kuang-Huei Lee, Xi Chen, Gang Hua, Houdong Hu, and Xiaodong He. Stacked cross attention for image-text matching. In ${ECCV}$ ,volume 11208,pages 212-228,2018. 15

[25] Yale Song and Mohammad Soleymani. Polysemous visual-semantic embedding for cross-modal retrieval. In CVPR, pages 1979-1988, 2019.

[26] Kunpeng Li, Yulun Zhang, Kai Li, Yuanyuan Li, and Yun Fu. Visual semantic reasoning for image-text matching. In ${ICCV}$ ,pages 4653-4661,2019.

[27] Hao Li, Jingkuan Song, Lianli Gao, Pengpeng Zeng, Haonan Zhang, and Gongfu Li. A differentiable semantic metric approximation in probabilistic embedding for cross-modal retrieval. In NeurIPS, volume 35, pages 11934-11946, 2022.

[28] Haiwen Diao, Ying Zhang, Lin Ma, and Huchuan Lu. Similarity reasoning and filtration for image-text matching. In AAAI, pages 1218-1226, 2021. 15

[29] Xuefeng Liang, Longshan Yao, Xingyu Liu, and Ying Zhou. Tripartite: Tackle noisy labels by a more precise partition. CoRR, 2022. 15

[30] Kun Yi and Jianxin Wu. Probabilistic end-to-end noise correction for learning with noisy labels. In ${CVPR}$ ,pages 7017-7025,2019. 16

[31] Zhilu Zhang and Mert R. Sabuncu. Generalized cross entropy loss for training deep neural networks with noisy labels. In NeurIPS, pages 8792-8802, 2018. 16

[32] Aditya Krishna Menon, Brendan van Rooyen, Cheng Soon Ong, and Bob Williamson. Learning from corrupted binary labels via class-probability estimation. In ${ICML}$ ,volume 37,pages 125-134, 2015.

[33] Nagarajan Natarajan, Inderjit S. Dhillon, Pradeep Ravikumar, and Ambuj Tewari. Learning with noisy labels. In NeurIPS, pages 1196-1204, 2013.

[34] Giorgio Patrini, Alessandro Rozza, Aditya Krishna Menon, Richard Nock, and Lizhen Qu. Making deep neural networks robust to label noise: A loss correction approach. In CVPR, pages 2233-2241, 2017.

[35] Xiaobo Xia, Tongliang Liu, Nannan Wang, Bo Han, Chen Gong, Gang Niu, and Masashi Sugiyama. Are anchor points really indispensable in label-noise learning? In NeurIPS, pages 6835-6846, 2019.

[36] Aritra Ghosh, Himanshu Kumar, and P. S. Sastry. Robust loss functions under label noise for deep neural networks. In Proceedings of the Thirty-First AAAI Conference on Artificial Intelligence, February 4-9, 2017, San Francisco, California, USA, pages 1919-1925. AAAI, 2017.

[37] Xinshao Wang, Yang Hua, Elyor Kodirov, David A Clifton, and Neil M Robertson. Imae for noise-robust learning: Mean absolute error does not treat examples equally and gradient magnitude's variance matters. arXiv preprint arXiv:1903.12141, 2019.

[38] Yisen Wang, Xingjun Ma, Zaiyi Chen, Yuan Luo, Jinfeng Yi, and James Bailey. Symmetric cross entropy for robust learning with noisy labels. In ICCV, pages 322-330, 2019.

[39] Yilun Xu, Peng Cao, Yuqing Kong, and Yizhou Wang. L_dmi: A novel information-theoretic loss function for training deep nets robust to label noise. In NeurIPS, pages 6222-6233, 2019.

[40] Zhilu Zhang and Mert R. Sabuncu. Generalized cross entropy loss for training deep neural networks with noisy labels. In NeurIPS, pages 8792-8802, 2018. 16

[41] Zeren Sun, Fumin Shen, Dan Huang, Qiong Wang, Xiangbo Shu, Yazhou Yao, and Jinhui Tang. PNP: robust learning from noisy labels by probabilistic noise prediction. In CVPR, pages 5301-5310, 2022. 16

[42] Paul Albert, Eric Arazo, Tarun Krishna, Noel E. O'Connor, and Kevin McGuinness. Is your noise correction noisy? PLS: robustness to label noise with two stage detection. In WACV, pages 118-127, 2023.

[43] Yazhou Yao, Zeren Sun, Chuanyi Zhang, Fumin Shen, Qi Wu, Jian Zhang, and Zhenmin Tang. Jo-src: A contrastive approach for combating noisy labels. In CVPR, pages 5192-5201, 2021. 16

[44] Paul Albert, Diego Ortego, Eric Arazo, Noel E. O'Connor, and Kevin McGuinness. Addressing out-of-distribution label noise in webly-labelled data. In WACV, pages 2393-2402, 2022. 16

[45] Dong Wang and Xiaoyang Tan. Robust distance metric learning via bayesian inference. IEEE Trans. Image Process., 27(3):1542-1553, 2018. 16

[46] Xinlong Yang, Haixin Wang, Jinan Sun, Shikun Zhang, Chong Chen, Xian-Sheng Hua, and Xiao Luo. Prototypical mixing and retrieval-based refinement for label noise-resistant image retrieval. In ${ICCV}$ ,pages ${11205} - {11215},{2023.16}$

[47] Sarah Ibrahimi, Arnaud Sors, Rafael Sampaio de Rezende, and Stéphane Clinchant. Learning with label noise for image retrieval by selecting interactions. In WACV, pages 468-477, 2022. 16
