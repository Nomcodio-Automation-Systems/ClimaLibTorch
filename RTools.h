#pragma once
#include <torch/torch.h>
#include "ClimaNet.h"
class RTools
{
	public:
	RTools();
	~RTools();
	static torch::Tensor r2_score(const torch::Tensor& input, const torch::Tensor& target, const std::string& multioutput = "uniform_average", int64_t num_regressors = 0);
	static torch::Tensor r2_dd_loss(const torch::Tensor& input, const torch::Tensor& target, double dynamic = 2.0, bool size_averag = true);

	static torch::Tensor divergence( torch::Tensor x, torch::Tensor y);

};

