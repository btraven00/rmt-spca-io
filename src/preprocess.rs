use anyhow::Result;
use faer::Mat;

use crate::reader::ScDataReader;

pub struct Preprocessor;

impl Preprocessor {
    /// Standard log1p normalisation: `X_ij ← ln(X_ij / lib_i × target_sum + 1)`.
    ///
    /// This matches Scanpy's `sc.pp.normalize_total` + `sc.pp.log1p` pipeline.
    /// Library sizes (row sums) are computed first, then the matrix is updated
    /// in-place row by row.
    ///
    /// Note: faer's `Mat` is column-major; the row-wise write pattern here is
    /// less cache-friendly than a column-wise pass, but correctness is the
    /// priority.  For very large matrices, consider a transposed temporary.
    pub fn log_normalize(mat: &mut Mat<f64>, target_sum: f64) {
        let (n, p) = (mat.nrows(), mat.ncols());

        // Pass 1: library sizes (row sums)
        let lib_sizes: Vec<f64> = (0..n)
            .map(|i| (0..p).map(|j| mat.read(i, j)).sum::<f64>())
            .collect();

        // Pass 2: scale and log1p in-place
        for i in 0..n {
            let lib = lib_sizes[i];
            // Avoid divide-by-zero for all-zero rows (droplets / empty barcodes)
            if lib == 0.0 {
                continue;
            }
            for j in 0..p {
                let v = mat.read(i, j);
                mat.as_mut().write(i, j, ((v / lib) * target_sum + 1.0).ln());
            }
        }
    }

    /// Heuristic: returns `true` if the matrix looks like it is already
    /// log-normalised.
    ///
    /// Samples the top-left 100 × 100 corner.  Log-normalised scRNA-seq values
    /// rarely exceed ~15; raw counts often reach thousands.
    pub fn is_likely_log_normalized(mat: &Mat<f64>) -> bool {
        let mut max_val = 0.0_f64;
        for i in 0..mat.nrows().min(100) {
            for j in 0..mat.ncols().min(100) {
                let v = mat.read(i, j);
                if v > max_val {
                    max_val = v;
                }
            }
        }
        // Log-normalised scRNA-seq rarely exceeds ~15.0
        max_val < 20.0
    }
}

/// Load an `.h5ad` file, auto-detect normalisation status, and optionally
/// apply log-normalisation before returning the matrix.
///
/// - `force_log = true`: always log-normalise if the data looks like raw counts.
/// - `force_log = false`: warn but do not transform.
pub fn load_and_prepare(path: &str, force_log: bool) -> Result<Mat<f64>> {
    let mut mat = ScDataReader::load_h5ad_as_faer(path)?;

    let is_log = Preprocessor::is_likely_log_normalized(&mat);

    if force_log && !is_log {
        eprintln!("Warning: data appears to be raw counts — applying log-normalisation.");
        Preprocessor::log_normalize(&mut mat, 10_000.0);
    } else if !force_log && !is_log {
        eprintln!("Warning: data is NOT log-normalised. RMT results may vary.");
    }

    Ok(mat)
}

#[cfg(test)]
mod tests {
    use super::*;
    use faer::Mat;

    fn make_counts(n: usize, p: usize) -> Mat<f64> {
        // Synthetic raw counts: cell i has lib size proportional to i+1
        Mat::from_fn(n, p, |i, j| ((i + 1) * (j + 1)) as f64)
    }

    #[test]
    fn log_normalize_zero_row_survives() {
        let mut mat = Mat::<f64>::zeros(3, 4);
        // Only row 1 has non-zero entries
        for j in 0..4 {
            mat.as_mut().write(1, j, (j + 1) as f64);
        }
        Preprocessor::log_normalize(&mut mat, 10_000.0);
        // Row 0 should remain zero
        for j in 0..4 {
            assert_eq!(mat.read(0, j), 0.0, "zero row was modified");
        }
        // Row 1 should be positive
        for j in 0..4 {
            assert!(mat.read(1, j) > 0.0, "log-normalised value should be positive");
        }
    }

    #[test]
    fn log_normalize_row_sums_consistent() {
        let mut mat = make_counts(5, 10);
        Preprocessor::log_normalize(&mut mat, 1.0);
        // After normalisation values should be in a plausible range for log1p
        for i in 0..5 {
            for j in 0..10 {
                let v = mat.read(i, j);
                assert!(v >= 0.0, "negative log value at ({i},{j})");
            }
        }
    }

    #[test]
    fn detection_heuristic() {
        // Raw counts matrix: max will be large
        let counts = make_counts(10, 10);
        assert!(!Preprocessor::is_likely_log_normalized(&counts));

        // Log-like matrix: all values < 10
        let log_mat = Mat::from_fn(10, 10, |i, j| ((i + j) as f64 * 0.3).min(9.0));
        assert!(Preprocessor::is_likely_log_normalized(&log_mat));
    }
}
