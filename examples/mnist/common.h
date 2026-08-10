#ifndef INSIGHTPP_COMMON_H
#define INSIGHTPP_COMMON_H

#include <iostream>
#include <filesystem>
#include <vector>
#include <fstream>
#include <cstring>

namespace fs = std::filesystem;

namespace inpp {

class MNIST {

/* API */
public:
    explicit MNIST(const fs::path& abs_mnist_dir) noexcept(false);

    const std::vector<std::vector<unsigned char>>& get_train_imgs() noexcept { return train_imgs; }
    const std::vector<unsigned char>& get_train_lbls() noexcept { return train_labels; }
    const std::vector<std::vector<unsigned char>>& get_test_imgs() noexcept { return test_imgs; }
    const std::vector<unsigned char>& get_test_lbls() noexcept { return test_labels; }

/* Routines */
private:
    void read_imgs() noexcept(false);
    void read_lbls() noexcept(false);

    static void read_imgs_from_file(std::vector<std::vector<unsigned char>>& images, const fs::path& abs_path) noexcept(false);
    static void read_lbls_from_file(std::vector<unsigned char>& labels, const fs::path& abs_path) noexcept(false);

    static uint32_t read_u32_big_endian_from_stream(std::istream& stream) noexcept(false);

/* Member */
private:
    const fs::path abs_mnist_dir;
    const fs::path abs_train_img_path;
    const fs::path abs_train_lbl_path;
    const fs::path abs_test_img_path;
    const fs::path abs_test_lbl_path;

    std::vector<std::vector<unsigned char>> train_imgs;
    std::vector<unsigned char> train_labels;
    std::vector<std::vector<unsigned char>> test_imgs;
    std::vector<unsigned char> test_labels;

/* Constants */
private:
    static constexpr auto TRAIN_IMGS_NAME = "train-images.idx3-ubyte";
    static constexpr auto TRAIN_LABELS_NAME = "train-labels.idx1-ubyte";
    static constexpr auto TEST_IMGS_NAME = "t10k-images.idx3-ubyte";
    static constexpr auto TEST_LABELS_NAME = "t10k-labels.idx1-ubyte";

    static constexpr std::size_t EXPECTED_TRAIN_IMGS_COUNT = 60000;
    static constexpr std::size_t EXPECTED_TEST_IMGS_COUNT = 10000;

    static constexpr uint32_t IMG_MAGIC_NUM = 0x0803;
    static constexpr uint32_t LBL_MAGIC_NUM = 0x0801;

    static constexpr std::size_t BUFFER_SIZE = 4096;

/* Public Constants */
public:
    static constexpr std::size_t IMG_ROW_NUM = 28;
    static constexpr std::size_t IMG_COL_NUM = 28;
    static constexpr std::size_t NUM_LABELS = 10;

};

inline MNIST::MNIST(const fs::path& abs_mnist_dir) noexcept(false)
    : abs_mnist_dir(abs_mnist_dir)
    , abs_train_img_path(abs_mnist_dir / TRAIN_IMGS_NAME)
    , abs_train_lbl_path(abs_mnist_dir / TRAIN_LABELS_NAME)
    , abs_test_img_path(abs_mnist_dir / TEST_IMGS_NAME)
    , abs_test_lbl_path(abs_mnist_dir / TEST_LABELS_NAME)
{
    if (abs_mnist_dir.empty() || !fs::is_directory(abs_mnist_dir)) {
        throw std::runtime_error("Path is empty or not a directory!");
    }

    if (!fs::is_regular_file(abs_train_img_path)) {
        throw std::runtime_error(abs_train_img_path.string() + " is not a regular file!");
    }

    if (!fs::is_regular_file(abs_train_lbl_path)) {
        throw std::runtime_error(abs_train_lbl_path.string() + " is not a regular file!");
    }

    if (!fs::is_regular_file(abs_test_img_path)) {
        throw std::runtime_error(abs_test_img_path.string() + " is not a regular file!");
    }

    if (!fs::is_regular_file(abs_test_lbl_path)) {
        throw std::runtime_error(abs_test_lbl_path.string() + " is not a regular file!");
    }

    read_imgs();
    read_lbls();

    if (train_imgs.size() != EXPECTED_TRAIN_IMGS_COUNT) {
        throw std::runtime_error(
            "Could not read all train images from provided file:\n"
            + abs_train_img_path.string() + "\n"
            + "Actual size: " + std::to_string(train_imgs.size()) + " / expected size: " + std::to_string(EXPECTED_TRAIN_IMGS_COUNT)
        );
    }

    if (train_labels.size() != EXPECTED_TRAIN_IMGS_COUNT) {
        throw std::runtime_error(
            "Could not read all train labels from provided file:\n"
            + abs_train_lbl_path.string() + "\n"
            + "Actual size: " + std::to_string(train_labels.size()) + " / expected size: " + std::to_string(EXPECTED_TRAIN_IMGS_COUNT)
        );
    }

    if (test_imgs.size() != EXPECTED_TEST_IMGS_COUNT) {
        throw std::runtime_error(
            "Could not read all test images from provided file:\n"
            + abs_test_img_path.string() + "\n"
            + "Actual size: " + std::to_string(test_imgs.size()) + " / expected size: " + std::to_string(EXPECTED_TEST_IMGS_COUNT)
        );
    }

    if (test_labels.size() != EXPECTED_TEST_IMGS_COUNT) {
        throw std::runtime_error(
            "Could not read all test labels from provided file:\n"
            + abs_test_lbl_path.string() + "\n"
            + "Actual size: " + std::to_string(test_labels.size()) + " / expected size: " + std::to_string(EXPECTED_TEST_IMGS_COUNT)
        );
    }

    std::cout << "Loaded MNIST training and test data successfully!" << std::endl;
}

inline void MNIST::read_imgs() noexcept(false)
{
    read_imgs_from_file(train_imgs, abs_train_img_path);
    read_imgs_from_file(test_imgs, abs_test_img_path);
}

inline void MNIST::read_lbls() noexcept(false)
{
    read_lbls_from_file(train_labels, abs_train_lbl_path);
    read_lbls_from_file(test_labels, abs_test_lbl_path);
}

inline void MNIST::read_imgs_from_file(std::vector<std::vector<unsigned char>>& images, const fs::path& abs_path) noexcept(false)
{
    std::ifstream imgs_file(abs_path, std::ios::in | std::ios::binary);

    const auto magic_num = read_u32_big_endian_from_stream(imgs_file);
    const auto images_num = read_u32_big_endian_from_stream(imgs_file);
    const auto images_rows = read_u32_big_endian_from_stream(imgs_file);
    const auto images_cols = read_u32_big_endian_from_stream(imgs_file);

    if (magic_num != IMG_MAGIC_NUM) {
        throw std::runtime_error("Error reading file: " + abs_path.string() + "!\nMagic number does not match!");
    }

    const auto img_size = images_rows * images_cols;
    std::vector<unsigned char> imgs;
    imgs.resize(images_num * img_size);
    std::size_t offset = 0;

    while (offset < imgs.size()) {
        const auto to_read = std::min(BUFFER_SIZE, imgs.size() - offset);

        imgs_file.read(reinterpret_cast<char*>(imgs.data() + offset), static_cast<std::streamsize>(to_read));

        const auto bytes_read = imgs_file.gcount();
        if (bytes_read <= 0) {
            break;
        }

        offset += static_cast<std::size_t>(bytes_read);
    }

    images = std::vector(images_num, std::vector<unsigned char>(img_size));
    for (std::size_t i = 0; i < images_num; ++i) {
        memcpy(images[i].data(), &imgs[i * img_size], img_size);
    }
}

inline void MNIST::read_lbls_from_file(std::vector<unsigned char>& labels, const fs::path& abs_path) noexcept(false)
{
    std::ifstream labels_file(abs_path, std::ios::in | std::ios::binary);

    const auto magic_num = read_u32_big_endian_from_stream(labels_file);
    const auto labels_num = read_u32_big_endian_from_stream(labels_file);

    if (magic_num != LBL_MAGIC_NUM) {
        throw std::runtime_error("Error reading file: " + abs_path.string() + "!\nMagic number does not match!");
    }

    labels.resize(labels_num);

    std::size_t offset = 0;
    while (offset < labels_num) {
        const auto to_read = std::min(BUFFER_SIZE, labels_num - offset);

        labels_file.read(reinterpret_cast<char*>(labels.data() + offset), static_cast<std::streamsize>(to_read));

        const auto bytes_read = labels_file.gcount();
        if (bytes_read <= 0) {
            break;
        }

        offset += static_cast<std::size_t>(bytes_read);
    }
}

inline uint32_t MNIST::read_u32_big_endian_from_stream(std::istream& stream) noexcept(false)
{
    unsigned char bytes[4];
    stream.read(reinterpret_cast<char*>(bytes), 4);

    if (!stream) {
        throw std::runtime_error("Failed to read u32 from stream!");
    }

    return (static_cast<std::uint32_t>(bytes[0]) << 24) |
           (static_cast<std::uint32_t>(bytes[1]) << 16) |
           (static_cast<std::uint32_t>(bytes[2]) <<  8) |
            static_cast<std::uint32_t>(bytes[3]);
}


} // namespace inpp



#endif //INSIGHTPP_COMMON_H
