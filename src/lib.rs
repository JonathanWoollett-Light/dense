//! Simpler and denser than MNIST, a dense file is binary file of seqeuantial training examples and nothing else (example->label->example->label->etc.).
//! 
//! Example labels and datapoints within examples can be stored using from 1 to 8 bytes (`label_bytes` and `data_bytes` respectively).
use std::{
    fs::File,
    io::{Read, Write},
};

/// Reads from dense file.
///
/// Returns (data,labels)
/// ```
/// use ndarray::{Array2,array};
///
/// let data: Array2<usize> = array![[0, 0], [1, 0], [0, 1], [1, 1]];
/// let labels: Array2<usize> = array![[0], [1], [1], [0]];
///
/// dense::write("dense_reader",1,1,&data,&labels);
///
/// let (read_data,read_labels) = dense::read("dense_reader",2,1,1);
///
/// assert_eq!(read_data,data);
/// assert_eq!(read_labels,labels);
///
/// # std::fs::remove_file("dense_reader");
/// ```
pub fn read(
    path: &str,          // Path to file
    example_size: usize, // Number of data points in example (e.g. pixels)
    data_bytes: usize,   // Number of bytes in data point
    label_bytes: usize,  // Number of bytes in label
) -> (ndarray::Array2<usize>, ndarray::Array2<usize>) {
    // Check sizes
    if data_bytes == 0 || data_bytes > 8 {
        panic!("All data points become `usize`s. `data_bytes` must be >=1 and <=8");
    }
    if label_bytes == 0 || label_bytes > 8 {
        panic!("All labels become `usize`s. `label_bytes` must be >=1 and <=8");
    }
    // Read file
    let mut file = File::open(path).unwrap();
    let mut buffer: Vec<u8> = Vec::new();
    file.read_to_end(&mut buffer)
        .expect("Failed dense file read.");
    // Set sizes
    let example_bytes: usize = (example_size * data_bytes) + label_bytes;
    let examples = buffer.len() / example_bytes;
    // Allocate storage
    let mut labels: Vec<usize> = vec![usize::default(); examples];
    let mut data: Vec<Vec<usize>> = vec![vec![usize::default(); example_size]; examples];
    // Set data
    for (indx, example) in buffer.chunks_exact(example_bytes).enumerate() {
        labels[indx] = u8s_to_usize(&example[example_size..]);
        data[indx] = example[0..example.len() - label_bytes]
            .chunks_exact(data_bytes)
            .map(|chunk| u8s_to_usize(chunk))
            .collect();
    }

    return (
        ndarray::Array::from_shape_vec(
            (examples, example_size),
            data.into_iter().flatten().collect(),
        )
        .expect("Read dense data shape wrong (this should be impossible)."),
        ndarray::Array::from_shape_vec((examples, 1), labels)
            .expect("Read dense label shape wrong (this should be impossible)."),
    );

    // Casts `&[u8]` to `usize`.
    // big endian: 1st `u8` is smallest component.
    fn u8s_to_usize(given_bytes: &[u8]) -> usize {
        let mut bytes: [u8; 8] = [0; 8];
        // `.rev()` since 1st byte is smallest (and thus last byte in usize) and last byte is largest (and thus 1st byte is usize)
        for (indx, byte) in given_bytes.iter().rev().enumerate() {
            bytes[indx] = *byte;
        }
        usize::from_le_bytes(bytes)
    }
}
/// Write to dense file.
///
/// ```
/// use ndarray::{Array2,array};
///
/// let data: Array2<usize> = array![[0, 0], [1, 0], [0, 1], [1, 1]];
/// let labels: Array2<usize> = array![[0], [1], [1], [0]];
///
/// dense::write("dense_writer",1,1,&data,&labels);
///
/// let (read_data,read_labels) = dense::read("dense_writer",2,1,1);
///
/// assert_eq!(read_data,data);
/// assert_eq!(read_labels,labels);
///
/// # std::fs::remove_file("dense_writer");
/// ```
pub fn write(
    path: &str,                      // Path to file
    data_bytes: usize,               // Number of bytes in data point
    label_bytes: usize,              // Number of bytes in label
    data: &ndarray::Array2<usize>,   // Data
    labels: &ndarray::Array2<usize>, // Labels
) {
    let mut file = File::create(path).expect("Failed to create file.");
    for (example_data, example_label) in data
        .axis_iter(ndarray::Axis(0))
        .zip(labels.axis_iter(ndarray::Axis(0)))
    {
        file.write_all(
            &example_data
                .as_slice()
                .expect("Failed to convert write dense data to slice.")
                .into_iter()
                .flat_map(|x| usize_to_u8s(*x, data_bytes))
                .collect::<Vec<u8>>(),
        )
        .expect("Failed dense data write.");
        file.write_all(
            &example_label
                .as_slice()
                .expect("Fail to convert write dense labels to slice.")
                .into_iter()
                .flat_map(|x| usize_to_u8s(*x, label_bytes))
                .collect::<Vec<u8>>(),
        )
        .expect("Failed dense label write.");
    }
    // Casts `usize` to `[u8:8]`.
    // big endian: 1st `u8` is smallest component.
    fn usize_to_u8s(value: usize, bytes: usize) -> Vec<u8> {
        let mut data_bytes: [u8; 8] = value.to_le_bytes();
        data_bytes.reverse(); // flip from little endian to big endian
        return data_bytes[8 - bytes..8].to_vec(); // returns required slice size
    }
}
