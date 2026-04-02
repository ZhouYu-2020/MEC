## Stand-alone MEC & Non-MEC 策略解析

### 两者都是**固定baselines**（不是学习策略）

1. **Non-MEC** (action=3): 所有任务都在**本地设备**处理
   ```python
   act = [3 for _ in range(env.user_num)]
   ```
   - 永远不使用MEC卸载
   - 用来对比：本地处理 vs MEC卸载

2. **Stand-alone MEC** (action=6): 所有任务**完全卸载到MEC**
   ```python
   act = [6 for _ in range(env.user_num)]
   ```
   - 永远充分利用MEC
   - 用来对比：完全卸载 vs 学习最优策略

### 为什么它们都是极端策略？

这是**设计意图**——它们是baselines来：
- 展示极端情况下的性能
- 说明学习策略（Full/IQL）相对于baselines的改进幅度

### 为什么仍然有不同的结果（despite相同的action）？

即使action相同，不同的seed会导致不同的结果，因为系统的随机性：
- **任务到达时间** 随机
- **卸载延迟** 随机
- **计算时间** 随机  
- **队列动态** 随机

### 现状的问题

❌ 只有2个seed，无法充分展现这些随机性的影响
❌ 150个有效episode，样本量太小
→ 误差棒太小（标准差接近0）

### 改进方案

✅ 增加到1000 episodes + 3 seeds → **会看到更多变异**
