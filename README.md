# PointsConcaveHull
凹包计算工具，用于精准提取空间转录组和空间代谢组的交集部分。


# 计算组织形状的方法 
- `凸包`：稳定、快，但会把凹陷“抹平”，不够贴合。(测试可以)
- `α-shape（凹包`：能贴合凹陷，是“尽可能贴合数据”的主力。（直接掉包不好用，建议自己写实现方法）
- `栅格化+等值线`：如果点云很密且你想要“视觉上的外轮廓”（尤其有孔洞、裂缝），它非常好用。（测试可以且速度快）
- `KNN凹包`：另一种“可控贴合”的凹包实现方式，适合你想更几何地控制边界。（不咋好用,速度慢） 



## 1. 凸包：作为baseline  
- 适用：你只需要一个外框，允许不贴合凹陷。
- 优点：简单、快、稳定
- 缺点：凹进去的地方全被“拉直” 

**代码**  
```
import numpy as np
from scipy.spatial import ConvexHull

points = df[['x', 'y']].values
pts = points 

def convex_boundary(pts):
    """
    输入:
      pts: (N, 2) ndarray

    输出:
      boundary_pts: (M, 2)，按顺序围一圈（闭合）
      boundary_idx: (M,)，对应原始 pts 的索引
    """
    pts = np.asarray(pts, float)
    hull = ConvexHull(pts)
    idx = hull.vertices           # 已经按顺/逆时针排序
    boundary = pts[idx]
    # 闭合一下
    boundary = np.vstack([boundary, boundary[0]])
    return boundary, idx 


boundary, idx = convex_boundary(pts)

plt.figure(figsize=(4,2.5))
plt.scatter(pts[:,0], pts[:,1], s=2, alpha=0.3, label="points")
plt.plot(boundary[:,0], boundary[:,1], "r-", lw=1.5, label="tu  convex hull")
plt.gca().set_aspect("equal", adjustable="box")
plt.legend()
plt.show()

```

**结果**  
![alt text](imgs/1凸包.png) 

-----  
