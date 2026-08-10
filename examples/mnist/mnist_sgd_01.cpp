#include "common.h" // Common header for all MNIST examples

#include "insightpp/data/container.h"
#include "insightpp/nn/linear.h"
#include "insightpp/math/activations.h"
#include "insightpp/data/initialization.h"

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
    using LabelBatch = inpp::data::Tensor<ValueType, batch_size, OUTPUT_DIM>;
    using TrainBatch = inpp::data::Tensor<ValueType, batch_size, INPUT_DIM>;

    // Load MNIST data by passing the directory which contains both the unzipped training and test files (4 in total)
    inpp::MNIST mnist{argv[1]};

    // Convert [60000, 28*28] shaped train images to a vector holding (60000 / batch_size) batches of shape [batch_size, 28*28]
    const std::vector<std::vector<unsigned char>>& train_images = mnist.get_train_imgs();
    std::vector<std::unique_ptr<TrainBatch>> train_image_batches;
    for (std::size_t i = 0; i < train_images.size() / batch_size; ++i) {
        auto b = std::make_unique<TrainBatch>(TrainBatch::zeros());

        for (std::size_t j = 0; j < batch_size; ++j) {

            for (std::size_t k = 0; k < TrainBatch::shape[1]; ++k) {
                (*b)(j, k) = static_cast<TrainBatch::value_type>(train_images[i * batch_size + j][k]) * (static_cast<TrainBatch::value_type>(1.0) / static_cast<TrainBatch::value_type>(255.0));
            }

        }

        train_image_batches.emplace_back(std::move(b));
    }

    // Convert [60000] shaped train images to a vector holding (60000 / batch_size) batches of shape [batch_size, 10]
    const std::vector<unsigned char>& train_labels = mnist.get_train_lbls();
    std::vector<std::unique_ptr<LabelBatch>> train_label_batches;
    for (std::size_t i = 0; i < train_labels.size() / batch_size; ++i) {
        auto b = std::make_unique<LabelBatch>(LabelBatch::zeros());

        for (std::size_t j = 0; j < batch_size; ++j) {
            (*b)(j, train_labels[i * batch_size + j]) = static_cast<LabelBatch::value_type>(1.0);
        }

        train_label_batches.emplace_back(std::move(b));
    }

    // Convert [10000, 28*28] shaped test images to a vector holding (10000 / batch_size) batches of shape [batch_size, 28*28]
    const std::vector<std::vector<unsigned char>>& test_images = mnist.get_test_imgs();
    std::vector<std::unique_ptr<TrainBatch>> test_image_batches;
    for (std::size_t i = 0; i < test_images.size() / batch_size; ++i) {
        auto b = std::make_unique<TrainBatch>(TrainBatch::zeros());

        for (std::size_t j = 0; j < batch_size; ++j) {

            for (std::size_t k = 0; k < TrainBatch::shape[1]; ++k) {
                (*b)(j, k) = static_cast<TrainBatch::value_type>(test_images[i * batch_size + j][k]) * (static_cast<TrainBatch::value_type>(1.0) / static_cast<TrainBatch::value_type>(255.0));
            }

        }

        test_image_batches.emplace_back(std::move(b));
    }

    // Convert [10000] shaped test images to a vector holding (10000 / batch_size) batches of shape [batch_size, 10]
    const std::vector<unsigned char>& test_labels = mnist.get_test_lbls();
    std::vector<std::unique_ptr<LabelBatch>> test_label_batches;
    for (std::size_t i = 0; i < test_labels.size() / batch_size; ++i) {
        auto b = std::make_unique<LabelBatch>(LabelBatch::zeros());

        for (std::size_t j = 0; j < batch_size; ++j) {
            (*b)(j, test_labels[i * batch_size + j]) = static_cast<LabelBatch::value_type>(1.0);
        }

        test_label_batches.emplace_back(std::move(b));
    }

    // Declare linear layer / activation and pass learning rate and weight initialization
    inpp::nn::Linear<ValueType, batch_size, INPUT_DIM, 512, inpp::init::RandomFromTo<-0.1f, 0.1f>> l1{lr};
    inpp::math::ReLU<ValueType, batch_size, 512> a1;
    inpp::nn::Linear<ValueType, batch_size, 512, 256, inpp::init::RandomFromTo<-0.1f, 0.1f>> l2{lr};
    inpp::math::ReLU<ValueType, batch_size, 256> a2;
    inpp::nn::Linear<ValueType, batch_size, 256, 128, inpp::init::RandomFromTo<-0.1f, 0.1f>> l3{lr};
    inpp::math::ReLU<ValueType, batch_size, 128> a3;
    inpp::nn::Linear<ValueType, batch_size, 128, 64, inpp::init::RandomFromTo<-0.1f, 0.1f>> l4{lr};
    inpp::math::ReLU<ValueType, batch_size, 64> a4;
    inpp::nn::Linear<ValueType, batch_size, 64, OUTPUT_DIM, inpp::init::RandomFromTo<-0.1f, 0.1f>> l5{lr};

    // Declare criterion
    inpp::math::MSE<ValueType, batch_size, OUTPUT_DIM> criterion;

    // Epoch loop
    for (std::size_t epoch = 0; epoch < epochs; ++epoch) {
        auto epoch_loss = static_cast<ValueType>(0.0);

        // Training loop over all batches (drop last)
        for (std::size_t b_idx = 0; b_idx < train_image_batches.size(); ++b_idx) {
            const auto& batch = train_image_batches[b_idx];
            const auto& label_batch = train_label_batches[b_idx];

            // Forward pass
            const auto& l1_o = l1.forward(*batch);
            const auto& a1_o = a1.forward(*l1_o);
            const auto& l2_o = l2.forward(*a1_o);
            const auto& a2_o = a2.forward(*l2_o);
            const auto& l3_o = l3.forward(*a2_o);
            const auto& a3_o = a3.forward(*l3_o);
            const auto& l4_o = l4.forward(*a3_o);
            const auto& a4_o = a4.forward(*l4_o);
            const auto& l5_o = l5.forward(*a4_o);

            const auto loss = criterion.forward(*label_batch, *l5_o);
            epoch_loss += loss;

            // Backward pass
            const auto& d_z_mse = criterion.backward();

            const auto& d_z5 = l5.backward(*d_z_mse);
            const auto& d_a4 = a4.backward(*d_z5);
            const auto& d_z4 = l4.backward(*d_a4);
            const auto& d_a3 = a3.backward(*d_z4);
            const auto& d_z3 = l3.backward(*d_a3);
            const auto& d_a2 = a2.backward(*d_z3);
            const auto& d_z2 = l2.backward(*d_a2);
            const auto& d_a1 = a1.backward(*d_z2);
            l1.backward(*d_a1);

            // Update weights
            l1.update();
            l2.update();
            l3.update();
            l4.update();
            l5.update();
        }

        std::size_t correct_samples = 0;
        auto test_loss = static_cast<ValueType>(0.0);

        for (std::size_t b_idx = 0; b_idx < test_image_batches.size(); ++b_idx) {
            const auto& test_batch = test_image_batches[b_idx];
            const auto& test_label_batch = test_label_batches[b_idx];

            // Forward pass
            const auto& l1_o = l1.forward(*test_batch);
            const auto& a1_o = a1.forward(*l1_o);
            const auto& l2_o = l2.forward(*a1_o);
            const auto& a2_o = a2.forward(*l2_o);
            const auto& l3_o = l3.forward(*a2_o);
            const auto& a3_o = a3.forward(*l3_o);
            const auto& l4_o = l4.forward(*a3_o);
            const auto& a4_o = a4.forward(*l4_o);
            const auto& l5_o = l5.forward(*a4_o);

            const auto& correct_samples_b = inpp::init::is_correct_label(*test_label_batch, *l5_o);
            for (std::size_t c_idx = 0; c_idx < batch_size; ++c_idx) {
                if (correct_samples_b(c_idx) == static_cast<ValueType>(1.0)) {
                    correct_samples++;
                }
            }

            const auto t_loss = criterion.forward(*test_label_batch, *l5_o);
            test_loss += t_loss;
        }

        std::cout << "Epoch " << epoch << " with loss: " << epoch_loss / static_cast<ValueType>(batch_size * train_image_batches.size()) << std::endl;
        std::cout << "\tAccuracy: " << static_cast<ValueType>(correct_samples) / static_cast<ValueType>(test_images.size()) << std::endl;
        std::cout << "\tTest Loss: " << test_loss / static_cast<float>(batch_size * test_image_batches.size()) << std::endl;
    }

    std::cout << "Finished training!" << std::endl;
}