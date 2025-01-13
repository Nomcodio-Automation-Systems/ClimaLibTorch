#pragma once

#include <torch/torch.h>
#include <string>

// Options struct for training hyperparameters
struct Options {
	bool dont_stop_training = true;
	bool save_model = true;
	bool debug = false;
	double learning_rate = 0.0001;
	size_t train_batch_size = 29;
	size_t test_batch_size = 200;
	size_t iterations = 10000;
	size_t log_interval = 20;
	size_t num_workers = 2;
	// path must end in delimiter
	std::string dataset_path = "./dataset/";
	std::string info_file_Path = "info.txt";
	torch::DeviceType device = torch::kCPU;
};