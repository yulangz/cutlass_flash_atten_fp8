#pragma once

#include <cstddef>
#include <cstdint>
#include <torch/extension.h>
#include <vector>

#include "flash.h"

std::vector<torch::Tensor> flash_attention_v2_fp8(torch::Tensor q, torch::Tensor k,
              torch::Tensor v, torch::Tensor q_scale_per_head, torch::Tensor k_scale_per_head, 
              torch::Tensor v_scale_per_head, bool is_causal = false, float softmax_scale=1);

