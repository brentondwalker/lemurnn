#ifndef LEMURNN_H
#define LEMURNN_H

#include <torch/script.h>
#include <torch/torch.h>
#include <string>
#include <utility>

class LEmuRnn {
public:
    /**
     * @brief Constructor that loads the TorchScript model.
     * @param model_path Path to the .pt model file.
     * @param device The torch::Device to run the model on (e.g., torch::kCPU or torch::kCUDA).
     */
    LEmuRnn(const std::string& model_path, int num_layers, int hidden_size,
	        double capacity, double queue_size, bool is_lstm=false);

    /**
     * Set/change the device of the model.
     */
    void to(torch::Device device);

    /**
     * Use the GPU if available.
     */
    void useGPU();

    /**
     * The hidden state is initialized to zeros at instantiation.
     * After long breaks in traffic one may want to reset it to zeros.
     */
    void resetHiddenState();
  
    /**
     * Gives more direct access to the model's forward(). 
     * You can supply the hidden state.
     * predict() replaced this.  Use that.
     *
     * @param input_x The input tensor of shape (batch_size, seq_len, input_size).
     * @param input_hidden The hidden state tensor of shape (num_layers, batch_size, hidden_size).
     * @return A std::pair containing:
     * - first: The output tensor (batch_size, seq_len, output_size).
     * - second: The new hidden state tensor (num_layers, batch_size, hidden_size).
     */
    std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> forward(
        const torch::Tensor& input_x, 
        const torch::Tensor& input_hidden
    );

    /**
     * A struct to return from the predict() function.
     * Contains a packet handling prediction.
     */
    struct PacketAction {
        double latency_ms;
        bool drop;
    };
    
    /**
     * Take the most recent inter-packet time and packet size and
     * use the model to make a latency/drop prediciton.
     */
    PacketAction predict(double inter_packet_time_ms, double packet_size_kbyte);
  
private:
    // the model hyperparams
    int num_layers_;
    int hidden_size_;
    
    // if this model is an LSTM, it needs a hidden and cell state
    bool is_lstm_;
  
    // the hidden state tensor
    torch::Tensor hidden_;
    torch::Tensor cell_state_;
    
    // the hidden state in the case of LSTM
    std::tuple<torch::Tensor,torch::Tensor> lstm_hidden_;

    // capacity in units of [Kbit/ms]=[Mbit/s]
    double capacity_;
    
    // queue capacity in units of [KByte]
    double queue_size_;

    // The loaded TorchScript module
    torch::jit::script::Module module;

    // The device the model runs on
    torch::Device device_;
};

#endif // LEMURNN_H

