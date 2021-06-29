#include <iostream>
#include "neuralnetwork.h"

void run(const cv::Mat &amp, const cv::Mat &ang);

int main() {

    bool on_gpu = true;

    std::string amp_path = "/home/terbed/PROJECTS/DHM/dhm_dnn_libtorch/input_samples/amp_0043.png";
    std::string ang_path = "/home/terbed/PROJECTS/DHM/dhm_dnn_libtorch/input_samples/ang_0043.png";

    cv::Mat amp = cv::imread(amp_path, cv::IMREAD_GRAYSCALE);
    cv::Mat ang = cv::imread(ang_path, cv::IMREAD_GRAYSCALE);

//    cv::resize(amp, amp, cv::Size(), 0.5, 0.5);
//    cv::resize(ang, ang, cv::Size(), 0.5, 0.5);
//    cv::resize(amp, amp, cv::Size(), 2, 2);
//    cv::resize(ang, ang, cv::Size(), 2, 2);

    // Check if loaded properly
    if(amp.empty())
    {
        std::cerr << "Could not read image: " << amp_path << std::endl;
        return 1;
    }
    if(ang.empty())
    {
        std::cerr << "Could not read image: " << ang_path << std::endl;
        return 1;
    }

    // Show input images
    cv::imshow("AMP", amp);
    cv::imshow("ANG", ang);
    cv::imwrite("/home/terbed/PROJECTS/DHM/dhm_dnn_libtorch/outputs/input_amp.png", amp);
    cv::imwrite("/home/terbed/PROJECTS/DHM/dhm_dnn_libtorch/outputs/input_ang.png", ang);

    // Load and test neural network
    NeuralNetwork* nn;
    if(on_gpu)
    {
        //nn = new NeuralNetwork("/home/terbed/PROJECTS/DHM/dhm_dnn_libtorch/traced_models/BF-GAN_orig_universal.pt", on_gpu);
        nn = new NeuralNetwork("/home/terbed/PROJECTS/DHM/dhm_dnn_libtorch/traced_models/algasys_deplv3.pt", on_gpu);
    }
    else
    {
        nn = new NeuralNetwork("/home/terbed/PROJECTS/DHM/dhm_dnn_libtorch/traced_models/algasys_deplv3.pt", on_gpu);
        //nn = new NeuralNetwork("/home/terbed/PROJECTS/DHM/dhm_dnn_libtorch/traced_models/BF-GAN_orig_universal_cpu.pt", on_gpu);
        // nn = new NeuralNetwork("/home/terbed/CLionProjects/dhm_dnn/traced_models/BF-GAN_orig_laser617_cpu.pt", on_gpu);
    }

    nn->testNN({1, 2, 512, 512});
    std::cout << std::endl;

    // Construct input for NN
    at::Tensor input = nn->mat2tensor(amp, ang);
    std::cout << "Dimension of input: " << input.sizes() << std::endl;

    // Run inference
    cv::Mat output = nn->run_inference(input);
    cv::imshow("OUT", output);
    cv::imwrite("/home/terbed/PROJECTS/DHM/dhm_dnn_libtorch/outputs/output_amp.png", output);

    int k = cv::waitKey(0);

    return 0;
}
