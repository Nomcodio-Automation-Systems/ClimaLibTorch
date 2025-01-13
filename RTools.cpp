#include "RTools.h"

RTools::RTools() = default;

RTools::~RTools() = default;
/*
* Calculate the R2 score for a set of predictions
* @param input: The predicted values
* @param target: The true values
* @param multioutput: The strategy for calculating the R2 score
* @param num_regressors: The number of regressors in the model
* @return: The R2 score
*/
torch::Tensor RTools::r2_score(const torch::Tensor& input, const torch::Tensor& target, const std::string& multioutput, int64_t num_regressors)
{
    // Calculate the mean of the target values
    torch::Tensor mean_target = torch::mean(target, 0);

    // Calculate the total sum of squares
    torch::Tensor total_sum_squares = torch::sum(torch::pow(target - mean_target, 2), 0);

    // Calculate the residual sum of squares
    torch::Tensor residual_sum_squares = torch::sum(torch::pow(target - input, 2), 0);

    // Calculate the R2 score
    torch::Tensor r2 = 1 - (residual_sum_squares / total_sum_squares);

    if (multioutput == "uniform_average") {
        // Average the R2 scores with uniform weight
        r2 = torch::mean(r2);
    }
    else if (multioutput == "raw_values") {
        // Return the full set of R2 scores
        // No additional action needed
    }
    else if (multioutput == "variance_weighted") {
        // Weight the R2 scores by the variances of each individual output
        torch::Tensor output_variances = torch::var(target, 0);
        r2 = torch::mean(r2 * output_variances);
    }

    if (num_regressors > 0) {
        // Adjust the R2 score for the number of regressors
        int64_t n_samples = input.size(0);
        torch::Tensor adjusted_r2 = 1 - ((1 - r2) * (n_samples - 1) / (n_samples - num_regressors - 1));
        return adjusted_r2;
    }

    return r2;
}
/*
* Calculate the dynamic divergence loss for a set of predictions
* @param input: The predicted values
* @param target: The true values
* @param dynamic: The dynamic parameter for the loss function
* @param size_average: Whether to average the loss over the batch
*/ 
torch::Tensor RTools::r2_dd_loss(const torch::Tensor& input, const torch::Tensor& target, double dynamic, bool size_average)
{
    torch::Tensor r2 = r2_score(input, target);
    torch::Tensor dd_l = torch::abs(r2 - 1) + 1; // 1 + abs(r2 - 1) -> + 1
    torch::Tensor s = torch::abs(1 / (r2 - 2));//relu(1 / (r2 - 2)) -> relu(1/(1-2)) = relu(-1) = 0
    torch::Tensor m = dynamic - dynamic * s; // 1 - 1 * 0 = 1
    torch::Tensor dd_loss = torch::pow(dd_l, m) - 1;

   
    if (size_average) {
        torch::Tensor mean = torch::mean(dd_loss);
        return mean;
    }
    else {
        torch::Tensor sum = torch::sum(dd_loss);
        return sum;
    }
}
/*
* Calculate the divergence between two tensors
* @param x: The first tensor
* @param y: The second tensor
*/
torch::Tensor RTools::divergence(torch::Tensor x, torch::Tensor y) {
  
    torch::Tensor difference = torch::abs(x - y);
    torch::Tensor percentage_difference = (difference / y) * 100;
    int64_t num_items = percentage_difference.numel();
    return percentage_difference.sum() / num_items;
}
