| 类别 | 调度周期占比 | op 数 |
|---|---:|---:|
| 数据搬运（Slice/assign/Transpose/Gather/Concat/Pad/Reshape/Squeeze） | 74.3% | 15449 |
| 逐元素（Mul/Add/Sub/Neg/Sigmoid/Tanh/Reduce） | 18.0% | 6624 |
| 矩阵乘（FullyConnected） | 4.7% | 1455 |
| 卷积（Conv） | 2.6% | 779 |
| 其他 | 0.5% | 207 |

总周期 1529M，调度 job 数 24514
