#[cfg(test)]
mod tests {
    const MNIST_TEST: usize = 10000;
    #[test]
    fn mnist() {
        // Reads MNIST
        let training_labels = mnist_read::read_labels("tests/mnist/t10k-labels.idx1-ubyte");
        let training_data = mnist_read::read_data("tests/mnist/t10k-images.idx3-ubyte");

        // Castes to ndarray's
        let usize_labels: Vec<usize> = training_labels.into_iter().map(|l| l as usize).collect();
        let labels: ndarray::Array2<usize> =
            ndarray::Array::from_shape_vec((MNIST_TEST, 1), usize_labels).expect("Bad labels");
        let old_labels_shape = labels.shape();

        let usize_data: Vec<usize> = training_data.into_iter().map(|d| d as usize).collect();
        let data: ndarray::Array2<usize> =
            ndarray::Array::from_shape_vec((MNIST_TEST, 28 * 28), usize_data).expect("Bad data");
        let old_data_shape = data.shape();

        // Writes dense
        dense::write("dense_mnist", 1, 1, &data, &labels);

        // Reads metadata
        let metadata =
            std::fs::metadata("dense_mnist").expect("Couldn't get meta data on dense file.");
        // Checks size
        assert_eq!(metadata.len() as usize, ((28 * 28) + 1) * MNIST_TEST);

        // Reads dense
        let (new_data, new_labels) = dense::read("dense_mnist", 28 * 28, 1, 1);

        // Checks correct sizes
        assert_eq!(new_labels.shape(), old_labels_shape);
        assert_eq!(new_data.shape(), old_data_shape);

        // Removes files
        std::fs::remove_file("dense_mnist").expect("Couldn't remove dense file");
    }
}
