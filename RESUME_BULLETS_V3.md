# Retail Decision Analytics — Resume Bullets V3

All bullets below are GREEN in `RESUME_EVIDENCE_MATRIX.md`. Select one role and one version; do not stack all bullets.

## A. 最小改动版

### 数据分析

- 分析 2,595,732 条零售交易，构建 RFM 客群、活动响应、品类关联与 8 周需求预测；Seasonal Naive 在留出集取得 9.30% WAPE。
- 仅使用活动前销售、频次和 recency 构建客户价值层级，观察响应率由低层级 7.78% 上升至高层级 16.65%，并以家庭聚类 bootstrap 给出区间。

### 商业分析

- 将客户价值与活动响应连接为触达优先级：group-safe 模型 ROC-AUC 0.826，最高评分十分位观察响应 lift 3.79×，不将排序结果解释为 promotion uplift。
- 量化四类客群的品类结构差异，识别 PASTRY、NUTRITION、KIOSK-GAS 与 MEAT-PCKGD 的 1.26–1.47× over-index，为品类测试提供假设。

### 数据产品

- 设计“客户—价值/行为—活动/品类响应—决策建议”分析流程，统一输出数据质量、客户画像、响应排序、品类指数和预测证据。
- 为决策模块定义发布边界：价值层级和品类指数只作观察性画像，活动模型只作 response ranking，复杂预测必须优于季节基线。

### AI 产品

- 将 260 万条交易转化为触达、品类和需求计划三类产品决策，使用 top-decile lift、品类 index 与 OOS WAPE 定义可验证成功指标。
- 基于高价值层级 16.65% 观察响应和客群品类差异提出定向实验假设，同时要求随机对照后才能判断增量效果。

### Data Science

- 在家庭级分组留出下训练活动响应模型，取得 ROC-AUC 0.826 与 3.79× top-decile observed lift，避免同一家庭跨训练/测试集泄漏。
- 构建活动前价值评分并对重复曝光按家庭进行 500 次 cluster bootstrap；高价值层级响应率 16.65%，95% CI 14.50%–18.97%。

## B. 最佳重写版

### 数据分析

- 将 259 万条交易明细重构为客户、活动、购物篮、品类与周度需求分析层，形成从 grain QA 到决策 KPI 的可复现证据链。
- 发现活动前客户价值与观察响应呈 7.78%→12.57%→16.65% 梯度，并用家庭聚类 bootstrap 处理重复曝光，为触达分层提供稳健描述证据。

### 商业分析

- 用 household-grouped 响应评分将活动受众排序，AUC 0.826、top-decile observed lift 3.79×，把有限营销资源优先投向高响应客户。
- 将四类客户画像延伸到商品结构：高价值、休眠、促销驱动等客群在重点品类上呈 1.26–1.47× over-index，转化为差异化品类实验建议。

### 数据产品

- 构建可审计零售决策原型，将 2.60M 交易贯通到客户价值、响应排序、品类差异、购物篮规则与需求基线五个模块。
- 把证据边界写入产品定义：区分 observed response、category association 与 causal uplift，并保留简单基线以阻止不必要的模型复杂化。

### AI 产品

- 从“触达谁、推荐什么、准备多少”定义用户决策和成功指标，建立 Customer → Value/Behavior → Response → Recommendation 产品链路。
- 以 16.65% 高价值层观察响应和 1.26–1.47× 品类指数生成实验 backlog；未随机化前只做优先级假设，不报告业务 uplift。

### Data Science

- 组合 pre-period feature engineering、group holdout 与 cluster bootstrap：活动模型 AUC 0.826，高价值层响应 16.65%（95% CI 14.50%–18.97%）。
- 用稳定性和样本外门槛约束模型选择：k=4 seed ARI 最低 0.962；Seasonal Naive WAPE 9.30% 并胜过 RF，避免复杂度导向。
