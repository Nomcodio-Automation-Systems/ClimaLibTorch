#pragma once
#include <torch/torch.h>
#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>


class ClimateDataset : public torch::data::Dataset<ClimateDataset> {
	using Data = std::vector<std::pair<torch::Tensor, torch::Tensor>>;
	using Example = torch::data::Example<>;
	Data data;
public:
	ClimateDataset(const Data& data) : data(data) {
		this->data = data;
	}



	torch::data::Example<> get(size_t index) override;

	torch::optional<size_t> size() const override;


	std::string csvFile_;
	std::string rootDir_;
	std::vector<std::pair<torch::Tensor, torch::Tensor>> data_;
};
