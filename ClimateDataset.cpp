#include "ClimateDataset.h"



    using Data = std::vector<std::pair<torch::Tensor, torch::Tensor>>;
    using Example = torch::data::Example<>;
    const Data data;
 
	/*
		This function loads the data from the CSV file
			  @pre: The CSV file must be in the format of:
			  x1, x2, x3, x4, y
			  @param size_t index: The index of the data to retrieve
			  @return: A vector of pairs of tensors containing the inputs and targets
			  @post: The data is loaded from the CSV file
	*/
    torch::data::Example<> ClimateDataset::get(size_t index)
    {
        auto inputs = data[index].first;
        auto targets = data[index].second;

        return { inputs.clone(), targets.clone() };
    }
	/*
		This function returns the size of the dataset
		@return: The size of the dataset
	*/
    torch::optional<size_t> ClimateDataset::size() const
    {
        return data.size();
    }
