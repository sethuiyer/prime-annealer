# Problem: Partition to K Sublists Minimizing Cross-Sum (Hard)

Given an array `nums` of `n` integers and an integer `k`, partition `nums` into `k` contiguous sublists such that the sum of products of max and min elements in each sublist is minimized.

Formally, find indices `0 = i0 < i1 < i2 < ... < ik = n` that minimize:

```
\sum_{t=1}^{k} (\max(nums[i_{t-1}..i_t-1]) * \min(nums[i_{t-1}..i_t-1]))
```

Return the minimal value of the objective.

**Constraints**
- `1 <= k <= n <= 200`
- `-10^4 <= nums[i] <= 10^4`

This is a hard DP/partitioning problem.
