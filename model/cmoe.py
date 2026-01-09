import torch
import torch.nn as nn
import torch.nn.functional as F


class MoE_cond(nn.Module):
    def __init__(self, input_dim, expert_dim, num_experts, k=3, num_generalists=1, cond_totlen=1024):
        super(MoE_cond, self).__init__()
        self.input_dim = input_dim
        self.expert_dim = expert_dim
        self.num_experts = num_experts
        self.k = k
        self.num_generalists = num_generalists  # 通才的数量

        self.experts = nn.ModuleList([nn.Linear(input_dim, expert_dim) for _ in range(num_experts)])
        self.generalist = nn.Linear(input_dim, expert_dim)

        self.gate = nn.Sequential(
            nn.Linear(cond_totlen, cond_totlen),
            nn.GELU(),
            nn.Linear(cond_totlen, num_experts)) 

    def forward(self, x, cond_flat):
        # 计算门控权重
        gate_scores = self.gate(cond_flat)
        gate_probs = F.softmax(gate_scores, dim=-1)

        # 选择 top-k 专家
        top_k_gate_probs, top_k_indices = torch.topk(gate_probs, self.k, dim=-1)

        expert_outputs = torch.stack([expert(x) for expert in self.experts], dim=1)  # shape: [batch_size, num_experts, expert_dim]

        # 根据 top_k_indices 提取出对应的专家输出，输出形状: [batch_size, k, expert_dim]
        top_k_expert_outputs = torch.gather(expert_outputs, dim=1, index=top_k_indices.unsqueeze(-1).expand(-1, -1, self.expert_dim))

        # 加权求和专家的输出，shape: [batch_size, k, expert_dim] * [batch_size, k] -> [batch_size, k, expert_dim]
        weighted_output = top_k_expert_outputs * top_k_gate_probs.unsqueeze(-1)

        # 求加权和，shape: [batch_size, expert_dim]
        output = weighted_output.sum(dim=1)

        # 加上通才的输出
        output = output + self.generalist(x)

        return output
