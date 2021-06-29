//
// Created by terbed on 2020. 12. 18..
//

#ifndef DHM_DNN_NEURALNETWORK_H
#define DHM_DNN_NEURALNETWORK_H

//#include "torch/torch.h"
#include "torch/script.h"
#include <vector>
#include <opencv2/opencv.hpp>
#include "c10/cuda/CUDAStream.h"

class NeuralNetwork
{
public:
    // inference API
    NeuralNetwork(const std::string& path_to_model, bool on_gpu);
    void testNN(at::IntArrayRef size);

    at::Tensor mat2tensor(const cv::Mat& amp, const cv::Mat& ang); /*!< convert Mat to Tensor */
    cv::Mat run_inference(at::Tensor x); /*!< run inference on input object x */
    bool on_gpu() const;

private:
    bool _on_gpu;
    torch::jit::script::Module _nn;
};

#endif //DHM_DNN_NEURALNETWORK_H
