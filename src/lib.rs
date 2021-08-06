//! An encoder/decoder to/from dense files.
//!
//! A file format simpler and denser than MNIST, a dense file is binary file of sequential training examples and nothing else (example->label->example->label->etc.).

use std::{
    fs::File,
    io::{Read, Write,BufWriter},
    mem,
};
use num_bytes::{IntoBytes,TryFromBytes};
use ndarray::{Axis,Array2};

/// Reads from dense file.
///
/// Returns (data,labels).
/// ```
/// use ndarray::{Array2,array};
///
/// let data: Array2<u8> = array![[0, 0], [1, 0], [0, 1], [1, 1]];
/// let labels: Array2<u16> = array![[0], [1], [1], [0]];
///
/// dense::write("dense_reader",&data,&labels).unwrap();
///
/// let (read_data,read_labels) = dense::read::<u8,u16>("dense_reader",2).unwrap();
///
/// assert_eq!(read_data,data);
/// assert_eq!(read_labels,labels);
///
/// # std::fs::remove_file("dense_reader").unwrap();
/// ```
pub fn read<T:TryFromBytes,P:TryFromBytes>(
    path: &str,          // Path to file
    example_size: usize, // Number of data points in each example (e.g. pixels in each image)
) -> Result<(Array2<T>, Array2<P>), Box<dyn std::error::Error>> {

    // Read file
    let mut file = File::open(path)?;
    let mut buffer: Vec<u8> = Vec::new();
    file.read_to_end(&mut buffer)?;

    let label_size = mem::size_of::<P>();
    let point_data_size = mem::size_of::<T>();
    let data_size = point_data_size * example_size;
    let sample_size = data_size+label_size;
    assert_eq!(buffer.len() % sample_size, 0);
    let length = buffer.len() / sample_size;

    let (data,labels): (Vec<Result<Vec<T>,_>>,Vec<Result<P,_>>) =  buffer.chunks_exact(sample_size).map(|chunk| {
        let temp_data: Result<Vec<T>,_> = chunk[0..data_size].chunks_exact(mem::size_of::<T>()).map(|c|{
            T::try_from_le_bytes(c)
        }).collect();
        debug_assert!(temp_data.is_ok());

        let label = P::try_from_le_bytes(&chunk[data_size..]);
        debug_assert!(label.is_ok());
        (temp_data,label)
    }).unzip();
    let clean_data: Vec<Vec<T>> = data.into_iter().collect::<Result<Vec<Vec<T>>,_>>()?;
    let clean_labels: Vec<P> = labels.into_iter().collect::<Result<Vec<P>,_>>()?;

    let flat_data: Vec<T> = clean_data.into_iter().flatten().collect();

    return Ok((
        ndarray::Array::from_shape_vec((length, example_size),flat_data)?,
        ndarray::Array::from_shape_vec((length, 1), clean_labels)?,
    ));
}
/// Writes to dense file.
///
/// ```
/// use ndarray::{Array2,array};
///
/// let data: Array2<u8> = array![[0, 0], [1, 0], [0, 1], [1, 1]];
/// let labels: Array2<u16> = array![[0], [1], [1], [0]];
///
/// dense::write("dense_writer",&data,&labels).unwrap();
///
/// let (read_data,read_labels) = dense::read::<u8,u16>("dense_writer",2).unwrap();
///
/// assert_eq!(read_data,data);
/// assert_eq!(read_labels,labels);
///
/// # std::fs::remove_file("dense_writer").unwrap();
/// ```
pub fn write<T: IntoBytes + Copy, P: IntoBytes + Copy>(
    path: &str,                      // Path to file
    data: &Array2<T>,   // Data
    labels: &Array2<P>, // Labels
) -> Result<(),Box<dyn std::error::Error>> {
    assert_eq!(data.len_of(Axis(0)), data.len_of(Axis(0)));

    let file = File::create(path)?;
    let mut writer = BufWriter::new(file);

    for (data_ndarray, label_ndarray) in data
        .axis_iter(Axis(0))
        .zip(labels.axis_iter(Axis(0)))
    { 
        let data_slice: &[T] = data_ndarray.as_slice().unwrap();
        let data_bytes: Vec<u8> = data_slice.iter().flat_map(|t|t.into_le_bytes()).collect();

        assert_eq!(label_ndarray.len(),1);
        let label: P = label_ndarray[0];
        let label_bytes = label.into_le_bytes();

        writer.write_all(&data_bytes)?;
        writer.write_all(&label_bytes)?;
    }
    Ok(())
}
