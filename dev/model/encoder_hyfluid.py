import torch
import numpy as np


class HashEncoderNative(torch.nn.Module):
    def __init__(
            self,
            num_levels: int = 16,
            min_res: int = 16,
            max_res: int = 256,
            log2_hashmap_size: int = 19,
            features_per_level: int = 2,
            hash_init_scale: float = 0.001,
            device=torch.device("cuda"),
    ):
        super().__init__()
        self.device = device
        self.num_levels = num_levels
        self.min_res = min_res
        self.max_res = max_res
        self.features_per_level = features_per_level

        # 用于哈希的素数
        self.primes = torch.tensor([1, 2654435761, 805459861, 3674653429], device=device, dtype=torch.int64)

        # 哈希表大小
        self.hash_table_size = 2 ** log2_hashmap_size

        # 根据 num_levels 计算分辨率增长因子和每层分辨率
        levels = torch.arange(self.num_levels, dtype=torch.float32)
        if self.num_levels > 1:
            self.growth_factor = np.exp((np.log(self.max_res) - np.log(self.min_res)) / (self.num_levels - 1))
        else:
            self.growth_factor = 1.0

        # 每个 level 的分辨率（保存在 GPU 上以便后续计算）
        self.scalings = torch.floor(
            self.min_res * (self.growth_factor ** levels)
        ).to(device, dtype=torch.float32)  # [L]

        # 每一层在哈希表中的 offset
        self.hash_offset = torch.arange(self.num_levels, dtype=torch.int64, device=device) * self.hash_table_size

        # 初始化哈希表参数
        hash_table = torch.rand(
            size=(self.hash_table_size * self.num_levels, self.features_per_level),
            device=device
        ) * 2 - 1  # [-1, 1]
        hash_table *= hash_init_scale
        self.hash_table = torch.nn.Parameter(hash_table)

        # 预先生成所有 16 个顶点在 4D 空间的二进制组合 (0或1)
        # shape: [16, 4]
        self.corner_offsets = torch.tensor(
            [[(i >> d) & 1 for d in range(4)] for i in range(16)],
            dtype=torch.int32,
            device=device
        )

    def hash_fn(self, in_tensor: torch.Tensor):
        """
        in_tensor: [N, L, 16, 4] (int32)
        返回: [N, L, 16] (int64)，哈希索引
        """
        # 转成 int64 与 primes 相乘，避免溢出
        in_tensor_64 = in_tensor.long()

        multiplied = in_tensor_64 * self.primes  # [N, L, 16, 4]
        x = multiplied[..., 0]
        x ^= multiplied[..., 1]
        x ^= multiplied[..., 2]
        x ^= multiplied[..., 3]

        # 对哈希表大小取模
        x %= self.hash_table_size

        # 加上每一层对应的 offset
        # x: [N, L, 16], hash_offset: [L]
        x = x + self.hash_offset.view(1, -1, 1)
        return x

    def _forward_chunk(self, xyzt_chunk: torch.Tensor):
        """
        处理一个相对小批量的 xyzt 输入（形状 [n, 4]）。
        返回 shape: [n, num_levels * features_per_level]
        """
        # 保证最后一维是 4
        # 扩展出 level 维度: xyzt_chunk: [n, 4] => [n, 1, 4]
        xyzt_chunk = xyzt_chunk.unsqueeze(-2)

        # [n, L, 4]
        scaled = xyzt_chunk * self.scalings.view(1, -1, 1)

        scaled_f = torch.floor(scaled).to(torch.int32)
        scaled_c = torch.ceil(scaled).to(torch.int32)
        offset = scaled - scaled_f.float()  # [n, L, 4], 小数部分

        # 构造 16 个角点坐标: [n, L, 16, 4]
        scaled_f_ = scaled_f.unsqueeze(2)  # [n, L, 1, 4]
        scaled_c_ = scaled_c.unsqueeze(2)  # [n, L, 1, 4]
        corner_offsets_ = self.corner_offsets.view(1, 1, 16, 4)  # [1, 1, 16, 4]

        corner_coords = torch.where(
            corner_offsets_ == 1,
            scaled_c_.expand(-1, -1, 16, -1),
            scaled_f_.expand(-1, -1, 16, -1)
        )  # [n, L, 16, 4]

        # 计算哈希表索引: [n, L, 16]
        corner_index = self.hash_fn(corner_coords)

        # 从哈希表中获取特征: corner_features => [n, L, 16, features_per_level]
        corner_index_flat = corner_index.view(-1)  # [n * L * 16]
        corner_features_flat = self.hash_table[corner_index_flat]  # [n * L * 16, F]
        corner_features = corner_features_flat.view(
            corner_index.shape[0], corner_index.shape[1], 16, self.features_per_level
        )

        # 计算 16 个顶点的插值权重
        # corner_offsets_[..., d] == 1 时用 offset[..., d]，否则用 (1 - offset[..., d])
        offset_ = offset.unsqueeze(2)  # [n, L, 1, 4]
        corner_offsets_float = corner_offsets_.float()  # [1, 1, 16, 4]
        corner_weight = torch.where(
            corner_offsets_float == 1.0,
            offset_,
            1.0 - offset_
        )  # [n, L, 16, 4]

        # 在最后一维(4)做乘积 => [n, L, 16]
        corner_weight = corner_weight.prod(dim=-1)

        # 将特征乘以对应权重并求和 => [n, L, F]
        interpolated = (corner_features * corner_weight.unsqueeze(-1)).sum(dim=2)

        # 展平 [L, F] => [L * F], 最终输出 [n, num_levels * features_per_level]
        out = interpolated.view(interpolated.shape[0], -1)
        return out

    def forward(self, xyzt: torch.Tensor, to_cpu=False, chunk_size=512 * 64):
        """
        对超大 xyzt 进行分批 (chunk) 处理。
        xyzt: 形状 [N, 4]（或更高维度，但最终展平到 [N, 4] 也行）
        chunk_size: 单次处理的数据量大小，可根据显存和需求调节
        """
        N = xyzt.shape[0]
        results = []

        start = 0
        while start < N:
            end = min(start + chunk_size, N)
            chunk = xyzt[start:end]  # [chunk_size, 4]
            out_chunk = self._forward_chunk(chunk)
            if to_cpu:
                out_chunk = out_chunk.cpu()
            results.append(out_chunk)
            start = end

        # 拼接所有 chunk 的结果
        return torch.cat(results, dim=0)
