#[cfg(test)]
mod tests {
    // Length of MNIST test data set
    const MNIST_TEST: usize = 10000;
    // Size of examples in MNIST
    const MNIST_SIZE: usize = 28*28;
    #[test]
    fn mnist() {
        // Reads MNIST
        let training_labels = mnist_read::read_labels("tests/mnist/t10k-labels.idx1-ubyte");
        let training_data = mnist_read::read_data("tests/mnist/t10k-images.idx3-ubyte");

        // Castes to `ndarray`s
        let labels = ndarray::Array::from_shape_vec((MNIST_TEST, 1), training_labels).unwrap();
        let data = ndarray::Array::from_shape_vec((MNIST_TEST, MNIST_SIZE), training_data).unwrap();

        // Writes dense
        dense::write("dense_mnist", &data, &labels).unwrap();

        // Reads metadata
        let metadata =
            std::fs::metadata("dense_mnist").unwrap();
        // Checks size
        assert_eq!(metadata.len() as usize, (MNIST_SIZE + 1) * MNIST_TEST);

        // Reads dense
        let (new_data, new_labels) = dense::read::<u8,u8>("dense_mnist", MNIST_SIZE).unwrap();

        // Checks correct sizes
        assert_eq!(new_labels.shape(), labels.shape());
        assert_eq!(new_data.shape(), data.shape());

        // Checks all data
        assert_eq!(new_data,data);
        assert_eq!(new_labels,labels);

        // Removes file
        std::fs::remove_file("dense_mnist").unwrap();
    }
}
