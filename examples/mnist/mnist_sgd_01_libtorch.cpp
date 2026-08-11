#include "common.h" // Common header for all MNIST examples

#include "torch/torch.h"

template <typename T>
constexpr torch::ScalarType torch_dtype();

template <>
constexpr torch::ScalarType torch_dtype<float>() { return torch::kFloat32; }

template <>
constexpr torch::ScalarType torch_dtype<double>() { return torch::kFloat64; }

int main(int argc, char* argv[])
{
// Hyperparameter
    static constexpr float lr = 0.512f;
    static constexpr std::size_t epochs = 100;
    static constexpr uint64_t batch_size = 512;

    static constexpr auto INPUT_DIM = inpp::MNIST::IMG_ROW_NUM * inpp::MNIST::IMG_COL_NUM;
    static constexpr auto OUTPUT_DIM = inpp::MNIST::NUM_LABELS;

    // Batch tensor types of insightpp
    using ValueType = float;
    using LabelBatch = torch::Tensor;
    using TrainBatch = torch::Tensor;

    // Load MNIST data by passing the directory which contains both the unzipped training and test files (4 in total)
    inpp::MNIST mnist{argv[1]};

    // Convert [60000, 28*28] shaped train images to a vector holding (60000 / batch_size) batches of shape [batch_size, 28*28]
    const std::vector<std::vector<unsigned char>>& train_images = mnist.get_train_imgs();
    std::vector<TrainBatch> train_image_batches;
    for (std::size_t i = 0; i < train_images.size() / batch_size; ++i) {
        auto b = torch::empty({batch_size, INPUT_DIM}, torch::TensorOptions().dtype(torch_dtype<ValueType>()));

        auto* dst = b.data_ptr<ValueType>();

        for (std::size_t j = 0; j < batch_size; ++j) {
            const auto& image = train_images[i * batch_size + j];

            for (std::size_t k = 0; k < INPUT_DIM; ++k) {
                dst[j * INPUT_DIM + k] = static_cast<ValueType>(image[k]) * (static_cast<ValueType>(1.0) / static_cast<ValueType>(255.0));
            }
        }

        train_image_batches.emplace_back(std::move(b));
    }

    // Convert [60000] shaped train images to a vector holding (60000 / batch_size) batches of shape [batch_size, 10]
    const std::vector<unsigned char>& train_labels = mnist.get_train_lbls();
    std::vector<LabelBatch> train_label_batches;
    for (std::size_t i = 0; i < train_labels.size() / batch_size; ++i) {
        auto b = torch::empty({batch_size, OUTPUT_DIM}, torch::TensorOptions().dtype(torch_dtype<ValueType>()));

        auto* dst = b.data_ptr<ValueType>();

        for (std::size_t j = 0; j < batch_size; ++j) {
            dst[j * OUTPUT_DIM + train_labels[i * batch_size + j]] = static_cast<ValueType>(1.0);
        }

        train_label_batches.emplace_back(std::move(b));
    }

    // Convert [10000, 28*28] shaped test images to a vector holding (10000 / batch_size) batches of shape [batch_size, 28*28]
    const std::vector<std::vector<unsigned char>>& test_images = mnist.get_test_imgs();
    std::vector<TrainBatch> test_image_batches;
    for (std::size_t i = 0; i < test_images.size() / batch_size; ++i) {
        auto b = torch::empty({batch_size, INPUT_DIM}, torch::TensorOptions().dtype(torch_dtype<ValueType>()));

        auto* dst = b.data_ptr<ValueType>();

        for (std::size_t j = 0; j < batch_size; ++j) {
            const auto& image = test_images[i * batch_size + j];

            for (std::size_t k = 0; k < INPUT_DIM; ++k) {
                dst[j * INPUT_DIM + k] = static_cast<ValueType>(image[k]) * (static_cast<ValueType>(1.0) / static_cast<ValueType>(255.0));
            }
        }

        test_image_batches.emplace_back(std::move(b));
    }

    // Convert [10000] shaped test images to a vector holding (10000 / batch_size) batches of shape [batch_size, 10]
    const std::vector<unsigned char>& test_labels = mnist.get_test_lbls();
    std::vector<LabelBatch> test_label_batches;
    for (std::size_t i = 0; i < test_labels.size() / batch_size; ++i) {
        auto b = torch::empty({batch_size, OUTPUT_DIM}, torch::TensorOptions().dtype(torch_dtype<ValueType>()));

        auto* dst = b.data_ptr<ValueType>();

        for (std::size_t j = 0; j < batch_size; ++j) {
            dst[j * OUTPUT_DIM + test_labels[i * batch_size + j]] = static_cast<ValueType>(1.0);
        }

        test_label_batches.emplace_back(std::move(b));
    }

    std::cout << train_label_batches[0] << std::endl;

    // Declare linear layer / activation to match mnist_sgd_01 example
    auto model = torch::nn::Sequential(
        torch::nn::Linear(INPUT_DIM, 512),
        torch::nn::ReLU(),
        torch::nn::Linear(512, 256),
        torch::nn::ReLU(),
        torch::nn::Linear(256, 128),
        torch::nn::ReLU(),
        torch::nn::Linear(128, 64),
        torch::nn::ReLU(),
        torch::nn::Linear(64, OUTPUT_DIM)
    );

    // Custom weight initialization to match mnist_sgd_01 example
    for (auto& l : model->children()) {
        if (auto linear = std::dynamic_pointer_cast<torch::nn::LinearImpl>(l)) {
            torch::nn::init::uniform_(linear->weight, -0.1, 0.1);
            torch::nn::init::uniform_(linear->bias, -0.1, 0.1);
        }
    }

    // Criterion to match mnist_sgd_01 example
    auto criterion = torch::nn::MSELoss(torch::nn::MSELossOptions().reduction(torch::kMean));
    torch::optim::SGD optimizer(model->parameters(), torch::optim::SGDOptions(lr).momentum(0)
        .dampening(0)
        .weight_decay(0)
        .nesterov(false)
    );

    for (std::size_t epoch = 0; epoch < epochs; ++epoch) {

        model->train();
        auto epoch_loss = static_cast<ValueType>(0.0);

        for (std::size_t b_idx = 0; b_idx < train_image_batches.size(); ++b_idx) {
            const auto& batch = train_image_batches[b_idx];
            const auto& label_batch = train_label_batches[b_idx];

            optimizer.zero_grad();

            const auto out = model->forward(batch);
            const auto loss = criterion->forward(out, label_batch);
            epoch_loss += loss.item<ValueType>();

            loss.backward();

            optimizer.step();
        }

        model->eval();

        std::size_t correct_samples = 0;
        auto test_loss = static_cast<ValueType>(0.0);

        torch::NoGradGuard no_grad;
        for (std::size_t b_idx = 0; b_idx < test_image_batches.size(); ++b_idx) {
            const auto& test_batch = test_image_batches[b_idx];
            const auto& test_label_batch = test_label_batches[b_idx];

            const auto out = model->forward(test_batch);
            const auto loss = criterion->forward(out, test_label_batch);

            test_loss += loss.item<ValueType>();

            const auto pred = out.argmax(1);
            const auto actual = test_label_batch.argmax(1);

            correct_samples += (pred == actual).sum().item<int64_t>();
        }

        std::cout << "Epoch " << epoch << " with loss: " << epoch_loss / static_cast<ValueType>(batch_size * train_image_batches.size()) << std::endl;
        std::cout << "\tAccuracy: " << static_cast<ValueType>(correct_samples) / static_cast<ValueType>(test_images.size()) << std::endl;
        std::cout << "\tTest Loss: " << test_loss / static_cast<float>(batch_size * test_image_batches.size()) << std::endl;
    }

    std::cout << "Finished training!" << std::endl;
}
