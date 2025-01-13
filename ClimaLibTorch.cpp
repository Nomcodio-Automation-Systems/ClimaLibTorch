#include <torch/torch.h>
#include <iostream>
#include <fstream>
#include <sstream>
#include <filesystem>
#include <string>
#include <vector>
#include "ClimateDataset.h"
#include "ClimaNet.h"
#include "Options.h"
#include "RTools.h"
#include <cmath>
#include <chrono>
#include "CSVLoadingException.h"
/*
	
*/

using Data = std::vector<std::pair<torch::Tensor, torch::Tensor>>;
Options options;
/*
* Load the data from the CSV file
* @pre: The CSV file must be in the format of:
*       x1, x2, x3, x4, y
* @return: A vector of pairs of tensors containing the inputs and targets
* 
*/
Data loadCSV() {
	Data data;
	std::ifstream file(options.info_file_Path);
	if (!file.is_open()) {
		throw std::runtime_error("Could not open file: " + options.info_file_Path);
	}

	// Skip the first line (header)
	std::string header;
	std::getline(file, header);

	std::string line;
	while (std::getline(file, line)) {
		std::stringstream ss(line);
		std::string cell;
		std::vector<float> row;

		// Skip the first two columns
		std::getline(ss, cell, ',');
		std::getline(ss, cell, ',');

		while (std::getline(ss, cell, ',')) {
			row.push_back(std::stof(cell));
		}

		if (row.size() > 1) {

			std::vector<float> inputs_data_start(row.begin(), row.begin() + 1);  // Use cells 1 inputs
			std::vector<float> inputs_data(row.begin() + 1, row.begin() + 5);  // Use cells 2, 3, 4 as input

			torch::Tensor inputs = torch::from_blob(inputs_data.data(), { 1, static_cast<int64_t>(inputs_data.size()) });

			torch::Tensor targets = torch::from_blob(inputs_data_start.data(), { 1, static_cast<int64_t>(inputs_data_start.size()) });

			data.emplace_back(inputs.clone(), targets.clone());
			// Print the inputs and targets for each data point
			//std::cout << "Inputs: " << inputs << std::endl;
			//std::cout << "Targets: " << targets << std::endl;
		}
	}

	file.close();
	return data;
}



// Helper function for runtime logging
void logRuntime(const std::chrono::high_resolution_clock::time_point& start, const std::string& output_file = "runtime.txt") {
	auto end = std::chrono::high_resolution_clock::now();
	auto duration_seconds = std::chrono::duration_cast<std::chrono::seconds>(end - start);
	auto duration_minutes = std::chrono::duration_cast<std::chrono::minutes>(duration_seconds);
	auto duration_hours = std::chrono::duration_cast<std::chrono::hours>(duration_minutes);

	auto remaining_minutes = duration_minutes - duration_hours;
	auto remaining_seconds = duration_seconds - duration_minutes;

	// Print the runtime in hours, minutes, and seconds
	std::cout << "Runtime: " << duration_hours.count() << " hours, "
		<< remaining_minutes.count() << " minutes, "
		<< remaining_seconds.count() << " seconds" << std::endl;

	// Write runtime to file
	std::ofstream outFile(output_file);
	if (outFile.is_open()) {
		outFile << "Runtime: " << duration_hours.count() << " hours, "
			<< remaining_minutes.count() << " minutes, "
			<< remaining_seconds.count() << " seconds" << std::endl;
		outFile.close();
	}
	else {
		std::cerr << "Error: Unable to open the output file: " << output_file << std::endl;
	}
}

int main() {
	auto start = std::chrono::high_resolution_clock::now();

	torch::manual_seed(1337);

	if (torch::cuda::is_available())
		options.device = torch::kCUDA;

	std::cout << "Using " << (options.device == torch::kCUDA ? "CUDA" : "CPU") << std::endl;

	std::filesystem::path current_path = std::filesystem::current_path();
	std::cout << "Current working directory: " << current_path << std::endl;

	auto net = ClimaNet(4, 50, 50, 50, 1);
	options.info_file_Path = R"(temp-co2csv8.csv)";
	options.learning_rate = 0.001;

	// Move the network and data onto the GPU if CUDA is available.
	net->to(options.device);

	auto data = loadCSV();
	auto train_set = ClimateDataset(data).map(torch::data::transforms::Stack<>());

	// Create a multi-threaded data loader for the dataset.
	auto train_loader = torch::data::make_data_loader<torch::data::samplers::RandomSampler>(
		std::move(train_set),
		torch::data::DataLoaderOptions().batch_size(options.train_batch_size).workers(options.num_workers));

	torch::optim::AdamOptions rate(options.learning_rate);
	bool stop_training = false;

	// Instantiate an optimization algorithm to update our Net's parameters.
	torch::optim::Adam optimizer(net->parameters(), rate);

	for (size_t epoch = 1; epoch <= options.iterations || options.dont_stop_training; epoch++) {
		size_t batch_index = 1;
		float r2 = 0.0;
		float div = 0.0;
		float closs = 0.0;

		for (auto& batch : *train_loader) {
			optimizer.zero_grad();

			auto inputs = batch.data.to(options.device);
			auto targets = batch.target.to(options.device);
			torch::Tensor prediction = net->forward(inputs);

			// Compute loss and related metrics
			torch::Tensor loss = RTools::r2_dd_loss(prediction, targets, 2).to(options.device);
			r2 += RTools::r2_score(prediction, targets).to(options.device).item<float>();
			div += RTools::divergence(prediction, targets).to(options.device).item<float>();
			closs += loss.clone().item<float>();

			loss.backward();
			optimizer.step();

			if (epoch % 100 == 0 && batch_index == 2) {
				if (options.debug) {
					std::cout << prediction << std::endl;
					std::cout << targets << std::endl;
				}

				r2 /= static_cast<float>(batch_index);
				div /= static_cast<float>(batch_index);
				closs /= static_cast<float>(batch_index);

				std::cout << "Epoch: " << epoch
					<< " | Batch: " << batch_index
					<< " | Loss: " << closs
					<< " | R2: " << r2
					<< " | divergence: " << div << "%" << std::endl;

				if (closs < 0.2 && !(div < 20.0)) {
					std::cout << "Divergence is too high! Stopping training." << std::endl;
					stop_training = true;
					break;
				}

				if (r2 > 0.90 && r2 <= 1 && (div < 10.0)) {
					if (options.save_model) {
						torch::save(net, "net.pt");
					}

					std::cout << "R2 is high enough! Stopping training." << std::endl;
					stop_training = true;
					break;
				}
			}
			batch_index++;
		}

		if (stop_training) {
			break;
		}
	}

	std::cout << "Training finished!" << std::endl;

	// Log runtime using helper function
	logRuntime(start);

	return 0;
}
