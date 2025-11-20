#include "lemurnn.h"
#include <iostream>
#include <vector>
#include <stdexcept>

LEmuRnn::LEmuRnn(const std::string& model_path, int num_layers, int hidden_size,
		 double capacity, double queue_size)
  : device_(torch::kCPU), num_layers_(num_layers), hidden_size_(hidden_size),
    capacity_(capacity), queue_size_(queue_size)
{    
    try {
        module = torch::jit::load(model_path);
        module.to(device_);
        module.eval();
        std::cout << "Model loaded from " << model_path
		  << " and moved to " << device_ << std::endl;
    }
    catch (const c10::Error& e) {
        std::cerr << "Error loading the model: " << e.what() << std::endl;
        throw std::runtime_error("Failed to load model");
    }

    // the initial hidden state
    int BATCH_SIZE = 1;
    hidden_ = torch::zeros({num_layers_, BATCH_SIZE, hidden_size_}).to(device_);
}

void LEmuRnn::resetHiddenState() {
    hidden_.zero_();
}

void LEmuRnn::useGPU() {
    if (torch::cuda::is_available()) {
        this->to(torch::kCUDA);
        std::cout << "Using CUDA (GPU)" << std::endl;
    } else {
        this->to(torch::kCPU);
        std::cout << "GPU not available.  Using CPU" << std::endl;
    }
}

void LEmuRnn::to(torch::Device device)
{
  device_ = device;
  module.to(device_);
}


/**
 * Direct access to the forward() function of the model.
 * If for some reason you want to use a different hidden state.
 */
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> LEmuRnn::forward(
    const torch::Tensor& input_x, 
    const torch::Tensor& input_hidden)
{
    std::vector<torch::jit::IValue> inputs;
    inputs.push_back(input_x.to(device_));
    inputs.push_back(input_hidden.to(device_));
    auto output_tuple = module.forward(inputs).toTuple();

    torch::Tensor backlog_out_tensor = output_tuple->elements()[0].toTensor();
    torch::Tensor drop_out_tensor = output_tuple->elements()[1].toTensor();
    torch::Tensor new_hidden_tensor = output_tuple->elements()[2].toTensor();

    return {backlog_out_tensor, drop_out_tensor, new_hidden_tensor};
}

/**
 * Primary function that will be called by a link emulator.
 * Returns a PacketAction prediction of how the packet should be handled.
 */
LEmuRnn::PacketAction LEmuRnn::predict(double inter_packet_time_ms, double packet_size_kbyte) {
    int BATCH_SIZE = 1;
    int SEQ_LEN = 1;
    int INPUT_SIZE = 4;
    // build the input vector
    torch::Tensor x = torch::zeros({BATCH_SIZE, SEQ_LEN, INPUT_SIZE});
    auto xa = x.accessor<float, 3>();
    xa[0][0][0] = inter_packet_time_ms;
    xa[0][0][1] = packet_size_kbyte;
    xa[0][0][2] = capacity_;
    xa[0][0][3] = queue_size_;
    
    // prepare inputs
    std::vector<torch::jit::IValue> inputs;
    inputs.push_back(x.to(device_));
    inputs.push_back(hidden_);

    auto output_tuple = module.forward(inputs).toTuple();
    torch::Tensor backlog_tensor = output_tuple->elements()[0].toTensor();
    torch::Tensor drop_tensor = output_tuple->elements()[1].toTensor();
    hidden_ = output_tuple->elements()[2].toTensor();

    // The raw output of the model is trained to be like a queue backlog.
    // Divide by capacity to get the corresponding latency.
    // [KB]/[KB/ms]=[ms]
    double latency_ms = backlog_tensor[0][0][0].item<double>() / capacity_;
    bool drop = (bool)torch::argmax(drop_tensor).item().toInt();
    //std::cout << "drop_tensor: " << drop_tensor << "\tdrop: " << drop << std::endl;
    PacketAction pa = {latency_ms, drop};
    return pa;
}

