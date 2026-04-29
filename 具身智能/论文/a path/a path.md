# A Path Towards Autonomous Machine Intelligence Version 0.9.2, 2022-06-27

Courant Institute of Mathematical Sciences, New York University yann@cs.nyu.edu Meta - Fundamental AI Research yann@fb.com

## Abstract
How could machines learn as efficiently as humans and animals? How could machines learn to reason and plan? How could machines learn representations of percepts and action plans at multiple levels of abstraction, enabling them to reason, predict, and plan at multiple time horizons? This position paper proposes an architecture and training paradigms with which to construct autonomous intelligent agents. It combines concepts such as configurable predictive world model, behavior driven through intrinsic motivation, and hierarchical joint embedding architectures trained with self-supervised learning.

Keywords: Artificial Intelligence, Machine Common Sense, Cognitive Architecture, Deep Learning, Self-Supervised Learning, Energy-Based Model, World Models, Joint Embedding Architecture, Intrinsic Motivation.

机器如何才能像人类和动物一样高效地学习？机器如何才能学会推理与规划？机器如何才能学习多层抽象层级下的感知表征与行动计划表征，从而具备在多时间跨度下开展推理、预测与规划的能力？本篇立场论文提出了一套用于构建自主智能体的架构与训练范式，融合了可配置预测型世界模型、内在动机驱动的行为模式，以及经自监督学习训练的分层联合嵌入架构等核心概念。

关键词：人工智能、机器常识、认知架构、深度学习、自监督学习、能量基模型、世界模型、联合嵌入架构、内在动机

## 1 Prologue

This document is not a technical nor scholarly paper in the traditional sense, but a position paper expressing my vision for a path towards intelligent machines that learn more like animals and humans, that can reason and plan, and whose behavior is driven by intrinsic objectives, rather than by hard-wired programs, external supervision, or external rewards. Many ideas described in this paper (almost all of them) have been formulated by many authors in various contexts in various form. The present piece does not claim priority for any of them but presents a proposal for how to assemble them into a consistent whole. In particular, the piece pinpoints the challenges ahead. It also lists a number of avenues that are likely or unlikely to succeed.


本文并非传统意义上的技术论文或学术论文，而是一篇立场论文，旨在阐述我对智能机器发展路径的愿景：这类智能机器能够以更贴近人类与动物的方式学习，具备推理和规划能力，其行为由内在目标驱动，而非硬编码程序、外部监督或外部奖励。
本文所阐述的诸多观点（几乎全部观点），此前已有众多研究者在不同场景下以不同形式提出过。本文并未宣称对其中任何观点享有首创优先权，而是提出了一套方案，阐释如何将这些观点整合为一个自洽的完整体系。尤为重要的是，本文精准点明了前路面临的核心挑战，同时也梳理了一系列有望落地与难以走通的研究路径。

The text is written with as little jargon as possible, and using as little mathematical prior knowledge as possible, so as to appeal to readers with a wide variety of backgrounds including neuroscience, cognitive science, and philosophy, in addition to machine learning, robotics, and other fields of engineering. I hope that this piece will help contextualize some of the research in AI whose relevance is sometimes difficult to see.

本文在撰写时尽量减少专业术语，也尽可能降低对数学基础的要求，以便让不同背景的读者都能理解，包括机器学习、机器人学及其他工程领域，同时也涵盖神经科学、认知科学与哲学等学科。我希望本文能帮助读者理解一些看似关联性不强的人工智能研究，厘清它们的定位与意义。

## 2 Introduction
Animals and humans exhibit learning abilities and understandings of the world that are far
beyond the capabilities of current AI and machine learning (ML) systems.


动物与人类所具备的学习能力和对世界的认知，远超当前人工智能（AI）与机器学习（ML）系统的能力上限。

How is it possible for an adolescent to learn to drive a car in about 20 hours of practice
and for children to learn language with what amounts to a small exposure. How is it that
most humans will know how to act in many situation they have never encountered? By
contrast, to be reliable, current ML systems need to be trained with very large numbers of
trials so that even the rarest combination of situations will be encountered frequently during
training. Still, our best ML systems are still very far from matching human reliability in
real-world tasks such as driving, even after being fed with enormous amounts of supervisory
data from human experts, after going through millions of reinforcement learning trials in
virtual environments, and after engineers have hardwired hundreds of behaviors into them.

青少年仅需约 20 小时练习就能学会开车，儿童只需少量语言接触便可掌握语言，这是如何做到的？为何大多数人即便面对从未经历过的场景，也知道该如何行事？
与之形成鲜明对比的是，当前机器学习系统要实现可靠运行，必须经过海量试错训练，才能让最罕见的场景组合也在训练中频繁出现。即便如此，当下最顶尖的机器学习系统，在自动驾驶等现实任务中的可靠性仍远不及人类 —— 即便我们为其输入海量人类专家的监督数据，让其在虚拟环境中完成数百万次强化学习试错，工程师还为其硬编码了数百种行为模式，也依然无法企及。

The answer may lie in the ability of humans and many animals to learn world models,
internal models of how the world works.

答案或许就在于人类和许多动物都具备学习世界模型的能力 —— 即构建关于世界运行规律的内部模型。

There are three main challenges that AI research must address today:
- 1. How can machines learn to represent the world, learn to predict, and learn to act
largely by observation?
Interactions in the real world are expensive and dangerous, intelligent agents should
learn as much as they can about the world without interaction (by observation) so
as to minimize the number of expensive and dangerous trials necessary to learn a
particular task.
- 2. How can machine reason and plan in ways that are compatible with gradient-based
learning?
Our best approaches to learning rely on estimating and using the gradient of a loss,
which can only be performed with differentiable architectures and is difficult to rec-
oncile with logic-based symbolic reasoning.
- 3. How can machines learn to represent percepts and action plans in a hierarchical man-
ner, at multiple levels of abstraction, and multiple time scales?
Humans and many animals are able to conceive multilevel abstractions with which
long-term predictions and long-term planning can be performed by decomposing com-
plex actions into sequences of lower-level ones.

当今人工智能研究必须攻克三大核心挑战：
- 1.机器如何主要依靠观察，完成世界表征学习、预测学习与行动学习？
现实世界中的交互成本高、风险大，智能体应尽可能通过无交互观察学习世界规律，以最小化完成特定任务所需的高成本、高风险试错次数。
- 2.机器如何以兼容基于梯度学习的方式开展推理与规划？
当前主流学习方法依赖损失梯度的计算与使用，这仅适用于可微架构，难以和基于逻辑的符号推理兼容。
- 3.机器如何以分层形式，在多层抽象、多时间尺度下学习感知表征与行动计划？
人类与许多动物能够构建多层级抽象表征，通过将复杂动作拆解为低层级动作序列，实现长时预测与长期规划。

The present piece proposes an architecture for intelligent agents with possible solutions to
all three challenges.

本文提出了一种智能体架构，可为上述三大挑战提供可行的解决方案。

The main contributions of this paper are the following:
- 1. an overall cognitive architecture in which all modules are differentiable and many of
them are trainable (Section 3, Figure 2).
- 2. JEPA and Hierarchical JEPA: a non-generative architecture for predictive world mod-
els that learn a hierarchy of representations (Sections 4.4 and 4.6, Figures 12 and 15).
- 3. a non-contrastive self-supervised learning paradigm that produces representations that
are simultaneously informative and predictable (Section 4.5, Figure 13).
- 4. A way to use H-JEPA as the basis of predictive world models for hierarchical planning
under uncertainty (section 4.7, Figure 16 and 17).

Impatient readers may prefer to jump directly to the aforementioned sections and figures.

本文的主要贡献如下：
- 1.提出一套完整的认知架构，架构中所有模块均可微，且多数模块具备可训练性（第 3 节，图 2）。
- 2.提出JEPA 与分层 JEPA（H-JEPA）：一种用于预测型世界模型的非生成式架构，可学习分层表征（第 4.4、4.6 节，图 12、图 15）。
- 3.提出一种非对比性自监督学习范式，能够生成兼具信息完整性与可预测性的表征（第 4.5 节，图 13）。
- 4.提出一种将H-JEPA作为预测型世界模型基础，用于不确定性环境下分层规划的方法（第 4.7 节，图 16、图 17）。

急于了解核心内容的读者，可直接跳转至上述章节与图表查看。

## 2.1 Learning World Models
Human and non-human animals seem able to learn enormous amounts of background knowl-
edge about how the world works through observation and through an incomprehensibly
small amount of interactions in a task-independent, unsupervised way. It can be hypoth-
esized that this accumulated knowledge may constitute the basis for what is often called
common sense. Common sense can be seen as a collection of models of the world that
can tell an agent what is likely, what is plausible, and what is impossible. Using such
world models, animals can learn new skills with very few trials. They can predict the con-
sequences of their actions, they can reason, plan, explore, and imagine new solutions to
problems. Importantly, they can also avoid making dangerous mistakes when facing an
unknown situation.

人类与其他动物似乎能够仅通过观察，以及极少的交互，以任务无关、无监督的方式，习得海量关于世界运行规律的背景知识。我们可以推测：这些积累而来的知识，正是人们常说的**常识**的基础。常识可被看作一组世界模型的集合，它能告诉智能体什么是大概率事件、什么是合理情况、什么是不可能发生的事。
借助这类世界模型，动物只需极少次尝试就能学会新技能。它们能预判自身行为的后果，能够推理、规划、探索，并构想解决问题的新方案。更重要的是，在面对未知情境时，它们还能避免犯下危险的错误。

The idea that humans, animals, and intelligent systems use world models goes back a
long time in psychology (Craik, 1943). The use of forward models that predict the next
state of the world as a function of the current state and the action being considered has been
standard procedure in optimal control since the 1950s (Bryson and Ho, 1969) and bears
the name model-predictive control. The use of differentiable world models in reinforcement
learning has long been neglected but is making a comeback (see for example (Levine, 2021))
A self-driving system for cars may require thousands of trials of reinforcement learning
to learn that driving too fast in a turn will result in a bad outcome, and to learn to slow
down to avoid skidding. By contrast, humans can draw on their intimate knowledge of
intuitive physics to predict such outcomes, and largely avoid fatal courses of action when
learning a new skill.

人类、动物及智能系统会运用**世界模型**的这一观点，在心理学领域由来已久（克雷克，1943）。**前向模型**（依据当前状态与拟执行动作，预测世界下一状态的模型）的应用，自 20 世纪 50 年代起便成为最优控制领域的标准方法（布莱森、何，1969），该方法也被称作**模型预测控制**。**可微世界模型**在强化学习中的应用曾长期被忽视，如今正重新成为研究热点（例如莱文，2021）。
汽车自动驾驶系统可能需要通过数千次强化学习试错，才能习得 “转弯时车速过快会引发不良后果” 这一规律，并学会减速以避免车辆侧滑。与之相反，人类能够凭借自身对直观物理规律的深刻认知预判此类结果，在学习新技能时便能从根本上规避危险的行为决策。

Common sense knowledge does not just allow animals to predict future outcomes, but
also to fill in missing information, whether temporally or spatially. It allows them to produce
interpretations of percepts that are consistent with common sense. When faced with an
ambiguous percept, common sense allows animals to dismiss interpretations that are not
consistent with their internal world model, and to pay special attention as it may indicate
a dangerous situation and an opportunity for learning a refined world model.
I submit that devising learning paradigms and architectures that would allow machines
to learn world models in an unsupervised (or self-supervised) fashion, and to use those
models to predict, to reason, and to plan is one of the main challenges of AI and ML today.
One major technical hurdle is how to devise trainable world models that can deal with
complex uncertainty in the predictions.

常识知识不仅能让动物预测未来结果，还能帮其填补时间或空间维度上的缺失信息。它能让动物对感知信息做出符合常识的解读。
当面对模糊的感知信号时，常识能让动物排除与内部世界模型相悖的解读，并对此类情况保持高度警惕 —— 这既可能预示危险，也是优化、完善世界模型的学习契机。
我认为，设计出能让机器以无监督（或自监督）方式学习世界模型，并利用这些模型开展预测、推理与规划的学习范式和架构，是当前人工智能与机器学习领域的核心挑战之一。其中一项关键技术难题，是如何构建可训练的世界模型，使其能够处理预测过程中的复杂不确定性。


## 2.2 Humans and Animals learn Hierarchies of Models
Humans and non-human animals learn basic knowledge about how the world works in the
first days, weeks, and months of life. Although enormous quantities of such knowledge are
acquired quite quickly, the knowledge seems so basic that we take it for granted. In the
first few months of life, we learn that the world is three-dimensional. We learn that every

## 2.2 人类与动物学习模型的层级结构
人类和其他动物在生命最初的几天、几周乃至几个月里，就会习得关于世界运行规律的基础知识。尽管这类知识的获取规模极大、速度极快，但它们太过基础，以至于我们对此习以为常。在出生后的最初几个月里，我们便认识到世界是三维的，我们会了解到每一个

<img width="2027" height="1135" alt="path1" src="https://github.com/user-attachments/assets/496de0bb-1895-4704-b7b1-5f2efbcef434" />

Figure 1: This chart, (courtesy of Emmanuel Dupoux), indicates at what age infants generally acquire various concepts about how the world works. It is consistent with the idea that abstract concepts, such as the fact that objects are subject to gravity and inertia, are acquired on top of less abstract concepts, like object permanence and the assignment of objects to broad categories. Much of this knowledge is acquired mostly by observation, with very little direct intervention, particularly in the first few weeks and months.

图 1：该图表（由伊曼纽尔・杜普沃提供）展示了婴儿通常在何种月龄习得各类关于世界运行规律的概念。这与下述观点相符：物体受重力与惯性作用等抽象概念，是在物体恒存性、物体泛化归类等更低抽象度概念的基础上逐步习得的。这类知识大多主要通过观察获取，几乎无需外界直接干预，这一点在出生后的最初几周、几个月里尤为明显。


source of light, sound, and touch in the world has a distance from us. The fact that every point in a visual percept has a distance is the best way to explain how our view of the world changes from our left eye to our right eye, or when our head is being moved. Parallax motion makes depth obvious, which in turn makes the notion of object obvious, as well as the fact that objects can occlude more distant ones. Once the existence of objects is established, they can be automatically assigned to broad categories as a function of their appearance or behavior. On top of the notion of object comes the knowledge that objects do not spontaneously appear, disappear, change shape, or teleport: they move smoothly and can only be in one place at any one time. Once such concepts are acquired, it becomes easy to learn that some objects are static, some have predictable trajectories (inanimate objects), some behave in somewhat unpredictable ways (collective phenomena like water, sand, tree leaves in the wind, etc), and some seem to obey different rules (animate objects). Notions of intuitive physics such as stability, gravity, inertia, and others can emerge on top of that. The effect of animate objects on the world (including the effects of the subject’s own actions) can be used to deduce cause-and-effect relationships, on top of which linguistic and social knowledge can be acquired.

世间所有光、声、触觉的来源，都与我们存在距离。视觉感知中每个点位都有远近深度，这一事实是解释左眼与右眼的视角差异、以及头部移动时我们对世界的观感会发生变化的最佳依据。

视差运动让深度感知变得直观，进而让 “物体” 的概念清晰起来，也让 “物体会遮挡更远物体” 的事实显而易见。一旦确立了物体的存在，大脑便会依据物体的外观或行为，将其自动归入宽泛的类别。
在 “物体” 概念之上，我们会进一步习得：物体不会凭空出现、消失、变形或瞬间转移；它们平稳移动，且同一时刻只能处于一个位置。掌握这些概念后，我们便能轻松区分：有些物体静止不动，有些运动轨迹可预测（非生物），有些行为难以预测（水、沙子、风中树叶等群体性现象），还有些遵循截然不同的规律（生物）。

在此基础上，稳定性、重力、惯性等直观物理概念便会逐步形成。生物对世界的作用（包括主体自身行为的影响）可用于推导因果关系，而语言与社会知识正是建立在这一因果认知之上。


Figure 1, courtesy of Emmanuel Dupoux, shows at what age infants seem to acquire
basic concepts such as object permanence, basic categories, intuitive physics, etc. Concepts
at higher levels of abstraction seem to develop on top of lower-level ones.

图 1（由伊曼纽尔・杜普沃提供）展示了婴儿大致在什么年龄段习得物体恒存性、基础类别、直观物理等基础概念。更高抽象层级的概念，似乎是在更低层级概念的基础上逐步发展形成的。

Equipped with this knowledge of the world, combined with simple hard-wired behav-
iors and intrinsic motivations/objectives, animals can quickly learn new tasks, predict the
consequences of their actions and plan ahead, foreseeing successful courses of actions and
avoiding dangerous situations.

凭借这些世界知识，再结合简单的先天固有行为与内在动机 / 目标，动物能够快速学习新任务、预判自身行为的后果并提前规划，预见可行的行动方案，同时规避危险情境。

But can a human or animal brain contain all the world models that are necessary for
survival? One hypothesis in this paper is that animals and humans have only one world
model engine somewhere in their prefrontal cortex. That world model engine is dynamically
configurable for the task at hand. With a single, configurable world model engine, rather
than a separate model for every situation, knowledge about how the world works may
be shared across tasks. This may enable reasoning by analogy, by applying the model
configured for one situation to another situation

但人类或动物的大脑，能否容纳生存所需的全部世界模型？本文提出一项假设：人类和动物的前额叶皮层中，仅存在一个世界模型引擎。该引擎可针对当前任务进行动态配置。相较于为每种场景单独构建模型，单一可配置的世界模型引擎能让世界运行规律的知识在不同任务间共享，进而实现类比推理—— 将适配某一场景的模型配置，迁移应用到另一新场景中。
