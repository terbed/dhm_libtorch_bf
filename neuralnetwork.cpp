//
// Created by terbed on 2020. 12. 18..
//

#include "neuralnetwork.h"

NeuralNetwork::NeuralNetwork(const std::string& path_to_model, bool on_gpu) : _on_gpu(on_gpu)
{
    try {
        // Deserialize the ScriptModule from a file using torch::jit::load().
        _nn = torch::jit::load(path_to_model);
    }  catch (const c10::Error& e) {
        std::cerr << "error loading the model\n" << std::endl;
        std::cerr << e.what() << "; " << e.msg() << std::endl;
    }

    std::cout << "Neural Network Succesfully Loaded!\n";

    if(on_gpu)
        _nn.to(torch::kCUDA);
    else
        _nn.to(torch::kCPU);

    // Put nn on selected device


    // Inference mode
    _nn.eval();
}

void NeuralNetwork::testNN(at::IntArrayRef size)
{
    torch::NoGradGuard no_grad;
    std::cout << "Testing network for random input...\n";

    for(int i=0; i<5; ++i)
    {
        std::vector<torch::jit::IValue> inputs;
        torch::Tensor input = torch::randn(size).set_requires_grad(false);

        if(_on_gpu)
            input = input.to(torch::kCUDA);
        else
            input = input.to(torch::kCPU);


        inputs.push_back(input);
        //inputs.push_back(torch::ones({1, 3, 128, 128, 128}).set_requires_grad(false));

        // Execute the model and turn its output into a tensor.
        auto start = std::chrono::system_clock::now();
        at::Tensor output = _nn.forward(inputs).toTensor();
        if(_on_gpu) // if on GPU wait until it finishes
        {
            at::cuda::CUDAStream stream = at::cuda::getCurrentCUDAStream();
            cudaStreamSynchronize(stream);
        }
        auto end = std::chrono::system_clock::now();
        auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
        std::cout << i << " | Running time for an input: " << elapsed << " ms\n";
    }

    std::cout << "Hurray!! Network is working!\n";


}

bool NeuralNetwork::on_gpu() const
{
    return _on_gpu;
}

at::Tensor NeuralNetwork::mat2tensor(const cv::Mat& amp, const cv::Mat& ang)
{
    int H = amp.rows;
    int W = amp.cols;

    auto start = std::chrono::system_clock::now();

    torch::NoGradGuard no_grad;

    // Convert to float
    cv::Mat amp_f, ang_f;
    amp.convertTo(amp_f, CV_32FC1, 1./255.);
    ang.convertTo(ang_f, CV_32FC1, 1./255.);

    // Convert image to tensor
    at::Tensor amp_tensor = torch::from_blob(amp_f.data, {amp_f.rows, amp_f.cols, amp_f.channels()}, torch::kF32).set_requires_grad(false);
    amp_tensor = amp_tensor.permute({ 2, 0, 1 }); // Channels x Height x Width

    at::Tensor ang_tensor = torch::from_blob(ang_f.data, {ang_f.rows, ang_f.cols, ang_f.channels()}, torch::kF32).set_requires_grad(false);
    ang_tensor = ang_tensor.permute({ 2, 0, 1 }); // Channels x Height x Width

    // concat channels
    at::Tensor input_tensor = torch::cat({amp_tensor, ang_tensor}, 0);

    // extract the spatial mean from each color-channels --> channel centralizaiton
    input_tensor = torch::sub(input_tensor, torch::mean(input_tensor, { 1, 2 }).reshape({ 2, 1 ,1 })).reshape({ 1, 2, H, W });

    auto end = std::chrono::system_clock::now();
    auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    std::cout << "Running time of the preprocessing: " << elapsed << " ms\n";

    return input_tensor;
}


cv::Mat NeuralNetwork::run_inference(at::Tensor x)
{
    if (_on_gpu)
        x = x.to(at::kCUDA);

    torch::NoGradGuard no_grad;

    std::vector<torch::jit::IValue> inputs;
    inputs.emplace_back(x);
    // Execute the model and turn its output into a tensor.
    auto start = std::chrono::system_clock::now();
    at::Tensor output = _nn.forward(inputs).toTensor();

    if(_on_gpu) // if on GPU wait until it finishes
    {
        at::cuda::CUDAStream stream = at::cuda::getCurrentCUDAStream();
        cudaStreamSynchronize(stream);
    }
    auto end = std::chrono::system_clock::now();
    auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    std::cout << "Running time of the Neural Network: " << elapsed << " ms\n";

    start = std::chrono::system_clock::now();
    // scale output back
    output = torch::add(output, 0.5);
    output = torch::mul(output, 255.);

    output = output.squeeze().to(at::kCPU);
    std::cout << "Output Size: " << output.sizes() << std::endl;


    // Load output into OpenCV matrix
    cv::Mat res(output.sizes()[0], output.sizes()[1], CV_32FC1, output.data_ptr()); // reinterpret_cast<uint8_t*>(
    res.convertTo(res, CV_8UC1);

    end = std::chrono::system_clock::now();
    elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    std::cout << "- Running time of the postprocessing: " << elapsed << " ms\n";

    return res;
}
