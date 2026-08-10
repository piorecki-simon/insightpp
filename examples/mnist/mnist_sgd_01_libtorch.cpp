#include "common.h" // Common header for all MNIST examples

#include "torch/torch.h"

class CustomMNIST : public torch::data::datasets::Dataset<CustomMNIST> {
public:
    explicit CustomMNIST(const inpp::MNIST& mnist_data) noexcept : mnist_data(mnist_data) {}

    torch::data::Example<> get(std::size_t index);

private:
    const inpp::MNIST& mnist_data;
};

int main(int argc, char* argv[])
{
// Hyperparameter
    static constexpr float lr = 0.512f;
    static constexpr std::size_t epochs = 100;
    static constexpr std::size_t batch_size = 512;

    static constexpr auto INPUT_DIM = inpp::MNIST::IMG_ROW_NUM * inpp::MNIST::IMG_COL_NUM;
    static constexpr auto OUTPUT_DIM = inpp::MNIST::NUM_LABELS;

    // Batch tensor types of insightpp
    using ValueType = float;

    // Load MNIST data by passing the directory which contains both the unzipped training and test files (4 in total)
    inpp::MNIST mnist{argv[1]};

    // Convert [60000, 28*28] shaped train images to a vector holding (60000 / batch_size) batches of shape [batch_size, 28*28]
    const std::vector<std::vector<unsigned char>>& train_images = mnist.get_train_imgs();

    // Convert [60000] shaped train images to a vector holding (60000 / batch_size) batches of shape [batch_size, 10]
    const std::vector<unsigned char>& train_labels = mnist.get_train_lbls();

    // Convert [10000, 28*28] shaped test images to a vector holding (10000 / batch_size) batches of shape [batch_size, 28*28]
    const std::vector<std::vector<unsigned char>>& test_images = mnist.get_test_imgs();

    // Convert [10000] shaped test images to a vector holding (10000 / batch_size) batches of shape [batch_size, 10]
    const std::vector<unsigned char>& test_labels = mnist.get_test_lbls();

}
