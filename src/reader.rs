use anyhow::{Context, Result};
use faer::Mat;
use ndarray::{Array2, ArrayD, Ix2};

use anndata::{ArrayElemOp, AnnDataOp, Backend};
use anndata::data::{ArrayData, DynArray, DynCsrMatrix, DynCscMatrix};
use anndata_hdf5::H5;
use nalgebra_sparse::CooMatrix;

pub struct ScDataReader;

impl ScDataReader {
    /// Open an `.h5ad` file (read-only) and return its `X` matrix as a dense
    /// `faer::Mat<f64>`.
    ///
    /// Handles dense (all numeric dtypes) and sparse CSR / CSC / non-canonical
    /// CSR storage formats, casting everything to f64.
    /// Read `var_names` (gene names) from an `.h5ad` file.
    /// Returns one string per column of X, in the same order.
    pub fn load_var_names(path: &str) -> Result<Vec<String>> {
        let file = H5::open(path).with_context(|| format!("cannot open HDF5 file: {path}"))?;
        let adata = anndata::AnnData::<H5>::open(file)
            .context("file is not a valid AnnData (.h5ad)")?;
        Ok(adata.var_names().into_vec())
    }

    pub fn load_h5ad_as_faer(path: &str) -> Result<Mat<f64>> {
        let file =
            H5::open(path).with_context(|| format!("cannot open HDF5 file: {path}"))?;
        let adata = anndata::AnnData::<H5>::open(file)
            .context("file is not a valid AnnData (.h5ad)")?;

        // get_x() → &ArrayElem<H5>;  ArrayElemOp::get::<ArrayData>() → Result<Option<ArrayData>>
        let data: ArrayData = adata
            .get_x()
            .get::<ArrayData>()
            .context("failed to read X from file")?
            .ok_or_else(|| anyhow::anyhow!("AnnData X slot is empty"))?;

        match data {
            ArrayData::Array(dyn_arr) => dense_dyn_to_faer(dyn_arr),
            ArrayData::CsrMatrix(dyn_csr) => Ok(sparse_csr_to_faer(dyn_csr)),
            ArrayData::CsrNonCanonical(nc) => {
                // canonicalize() merges duplicate entries → DynCsrMatrix
                let canonical = nc
                    .canonicalize()
                    .map_err(|_| anyhow::anyhow!("failed to canonicalize CsrNonCanonical X"))?;
                Ok(sparse_csr_to_faer(canonical))
            }
            ArrayData::CscMatrix(dyn_csc) => Ok(sparse_csc_to_faer(dyn_csc)),
            ArrayData::DataFrame(_) => {
                Err(anyhow::anyhow!("X is a DataFrame, not a numeric matrix"))
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Dense conversions
// ---------------------------------------------------------------------------

fn dense_dyn_to_faer(arr: DynArray) -> Result<Mat<f64>> {
    fn cast<T>(a: ArrayD<T>, f: impl Fn(T) -> f64) -> Result<Mat<f64>>
    where
        T: Clone,
    {
        let a2: Array2<T> = a
            .into_dimensionality::<Ix2>()
            .context("X dense array is not 2-D")?;
        Ok(dense_arr2_to_faer(&a2, f))
    }

    match arr {
        DynArray::F32(a) => cast(a, |v| v as f64),
        DynArray::F64(a) => cast(a, |v| v),
        DynArray::I8(a) => cast(a, |v| v as f64),
        DynArray::I16(a) => cast(a, |v| v as f64),
        DynArray::I32(a) => cast(a, |v| v as f64),
        DynArray::I64(a) => cast(a, |v| v as f64),
        DynArray::U8(a) => cast(a, |v| v as f64),
        DynArray::U16(a) => cast(a, |v| v as f64),
        DynArray::U32(a) => cast(a, |v| v as f64),
        DynArray::U64(a) => cast(a, |v| v as f64),
        DynArray::Usize(a) => cast(a, |v| v as f64),
        DynArray::Bool(_) | DynArray::String(_) | DynArray::Categorical(_) => {
            Err(anyhow::anyhow!("X has non-numeric element type"))
        }
    }
}

/// Fill a `Mat<f64>` from a typed `Array2<T>`, parallel over rows (ndarray is
/// row-major so this is cache-friendly on the read side).
fn dense_arr2_to_faer<T: Clone>(arr: &Array2<T>, cast: impl Fn(T) -> f64) -> Mat<f64> {
    let (n, p) = (arr.nrows(), arr.ncols());
    // Row-major flat buffer
    let mut flat = vec![0.0_f64; n * p];
    // Sequential fill — rayon would need T: Send+Sync; keep simple for now
    for i in 0..n {
        for j in 0..p {
            flat[i * p + j] = cast(arr[[i, j]].clone());
        }
    }
    Mat::from_fn(n, p, |i, j| flat[i * p + j])
}

// ---------------------------------------------------------------------------
// Sparse conversions
// ---------------------------------------------------------------------------

/// Materialise a `DynCsrMatrix` into a dense `Mat<f64>`.
fn sparse_csr_to_faer(dyn_csr: DynCsrMatrix) -> Mat<f64> {
    macro_rules! csr_mat {
        ($csr:expr, $cast:expr) => {{
            let (n, p) = ($csr.nrows(), $csr.ncols());
            let mut mat = Mat::<f64>::zeros(n, p);
            for (i, row) in $csr.row_iter().enumerate() {
                for (&j, &v) in row.col_indices().iter().zip(row.values()) {
                    mat.as_mut().write(i, j, $cast(v));
                }
            }
            mat
        }};
    }

    match dyn_csr {
        DynCsrMatrix::F32(m) => csr_mat!(m, |v: f32| v as f64),
        DynCsrMatrix::F64(m) => csr_mat!(m, |v: f64| v),
        DynCsrMatrix::I8(m) => csr_mat!(m, |v: i8| v as f64),
        DynCsrMatrix::I16(m) => csr_mat!(m, |v: i16| v as f64),
        DynCsrMatrix::I32(m) => csr_mat!(m, |v: i32| v as f64),
        DynCsrMatrix::I64(m) => csr_mat!(m, |v: i64| v as f64),
        DynCsrMatrix::U8(m) => csr_mat!(m, |v: u8| v as f64),
        DynCsrMatrix::U16(m) => csr_mat!(m, |v: u16| v as f64),
        DynCsrMatrix::U32(m) => csr_mat!(m, |v: u32| v as f64),
        DynCsrMatrix::U64(m) => csr_mat!(m, |v: u64| v as f64),
        DynCsrMatrix::Usize(m) => csr_mat!(m, |v: usize| v as f64),
        DynCsrMatrix::Bool(m) => csr_mat!(m, |v: bool| if v { 1.0 } else { 0.0 }),
        DynCsrMatrix::String(_) => {
            // String sparse X is not meaningful; return empty matrix
            Mat::zeros(0, 0)
        }
    }
}

/// Materialise a `DynCscMatrix` via COO round-trip into a dense `Mat<f64>`.
fn sparse_csc_to_faer(dyn_csc: DynCscMatrix) -> Mat<f64> {
    macro_rules! csc_mat {
        ($csc:expr, $cast:expr) => {{
            let coo: CooMatrix<_> = CooMatrix::from(&$csc);
            let (n, p) = (coo.nrows(), coo.ncols());
            let mut mat = Mat::<f64>::zeros(n, p);
            for (i, j, v) in coo.triplet_iter() {
                mat.as_mut().write(i, j, $cast(*v));
            }
            mat
        }};
    }

    use anndata::data::DynCscMatrix;
    match dyn_csc {
        DynCscMatrix::F32(m) => csc_mat!(m, |v: f32| v as f64),
        DynCscMatrix::F64(m) => csc_mat!(m, |v: f64| v),
        DynCscMatrix::I8(m) => csc_mat!(m, |v: i8| v as f64),
        DynCscMatrix::I16(m) => csc_mat!(m, |v: i16| v as f64),
        DynCscMatrix::I32(m) => csc_mat!(m, |v: i32| v as f64),
        DynCscMatrix::I64(m) => csc_mat!(m, |v: i64| v as f64),
        DynCscMatrix::U8(m) => csc_mat!(m, |v: u8| v as f64),
        DynCscMatrix::U16(m) => csc_mat!(m, |v: u16| v as f64),
        DynCscMatrix::U32(m) => csc_mat!(m, |v: u32| v as f64),
        DynCscMatrix::U64(m) => csc_mat!(m, |v: u64| v as f64),
        DynCscMatrix::Usize(m) => csc_mat!(m, |v: usize| v as f64),
        DynCscMatrix::Bool(m) => {
            // CooMatrix<bool> requires bool: Zero which isn't satisfied; materialise directly
            let (n, p) = (m.nrows(), m.ncols());
            let mut mat = Mat::<f64>::zeros(n, p);
            // Iterate via CSR view (nalgebra can convert CSC→CSR same type)
            let csr = nalgebra_sparse::CsrMatrix::from(&m);
            for (i, row) in csr.row_iter().enumerate() {
                for (&j, &v) in row.col_indices().iter().zip(row.values()) {
                    mat.as_mut().write(i, j, if v { 1.0 } else { 0.0 });
                }
            }
            mat
        }
        DynCscMatrix::String(_) => Mat::zeros(0, 0),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::arr2;

    #[test]
    fn dense_f32_roundtrip() {
        let a: Array2<f32> = arr2(&[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]);
        let m = dense_arr2_to_faer(&a, |v| v as f64);
        assert_eq!((m.nrows(), m.ncols()), (2, 3));
        assert!((m.read(0, 0) - 1.0).abs() < 1e-9);
        assert!((m.read(1, 2) - 6.0).abs() < 1e-9);
    }

    #[test]
    fn sparse_csr_u32_roundtrip() {
        // Raw integer counts: [[1, 0, 2], [0, 3, 0]]
        let mut coo = CooMatrix::<u32>::new(2, 3);
        coo.push(0, 0, 1);
        coo.push(0, 2, 2);
        coo.push(1, 1, 3);
        use nalgebra_sparse::CsrMatrix;
        let m = sparse_csr_to_faer(DynCsrMatrix::U32(CsrMatrix::from(&coo)));
        assert!((m.read(0, 0) - 1.0).abs() < 1e-9);
        assert!((m.read(0, 2) - 2.0).abs() < 1e-9);
        assert!((m.read(1, 1) - 3.0).abs() < 1e-9);
        assert_eq!(m.read(0, 1), 0.0);
    }
}
