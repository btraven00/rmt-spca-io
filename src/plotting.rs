use anyhow::{Context, Result};
use faer::Mat;
use plotters::prelude::*;
use pyo3::prelude::*;
use pyo3::types::PyDict;

// ── Palette ───────────────────────────────────────────────────────────────────

// ColorBrewer Set1 — 9 maximally-distinct hues for categorical data.
const PALETTE: &[RGBColor] = &[
    RGBColor(228,  26,  28), // red
    RGBColor( 55, 126, 184), // blue
    RGBColor( 77, 175,  74), // green
    RGBColor(152,  78, 163), // purple
    RGBColor(255, 127,   0), // orange
    RGBColor(166,  86,  40), // brown
    RGBColor(247, 129, 191), // pink
    RGBColor(153, 153, 153), // grey
    RGBColor(255, 255,  51), // yellow
];

// ── Cell-score projection ─────────────────────────────────────────────────────

/// Project `x` (n_cells × p_genes) onto `w` (p_genes × k_components).
/// Returns the (n_cells × k_components) score matrix.
/// Extra columns in `x` beyond `w.nrows()` are silently ignored.
pub fn project_cells(x: &Mat<f64>, w: &Mat<f64>) -> Mat<f64> {
    let n = x.nrows();
    let p = x.ncols().min(w.nrows());
    let k = w.ncols();
    Mat::from_fn(n, k, |i, c| {
        (0..p).map(|g| x.read(i, g) * w.read(g, c)).sum::<f64>()
    })
}

// ── Dominant-component colours ────────────────────────────────────────────────

fn dominant_components(scores: &Mat<f64>, noise_floor: f64) -> Vec<usize> {
    let (n, k) = (scores.nrows(), scores.ncols());
    (0..n)
        .map(|i| {
            let (best_c, best_v) = (0..k).fold((k, 0.0_f64), |(bc, bv), c| {
                let v = scores.read(i, c).abs();
                if v > bv { (c, v) } else { (bc, bv) }
            });
            if best_v >= noise_floor { best_c } else { k }
        })
        .collect()
}

fn component_color(c: usize, k: usize) -> RGBColor {
    if c >= k { RGBColor(210, 210, 210) } else { PALETTE[c % PALETTE.len()] }
}

// ── faer → contiguous f32 row-major buffer ────────────────────────────────────

fn faer_to_vec_f32(mat: &Mat<f64>) -> Vec<f32> {
    let (nrows, ncols) = (mat.nrows(), mat.ncols());
    let mut buf = vec![0.0f32; nrows * ncols];
    for j in 0..ncols {
        let col = mat.col(j);
        for i in 0..nrows { buf[i * ncols + j] = col[i] as f32; }
    }
    buf
}

// ── UMAP via umap-learn (Python) ──────────────────────────────────────────────

/// Call `umap.UMAP(n_neighbors, min_dist, init='spectral').fit_transform(data)`
/// through the embedded Python interpreter.
///
/// We pass the data as a plain `Vec<f32>` and let Python/numpy handle the
/// reshape — this avoids all Rust↔numpy binding version conflicts.
///
/// Requires `umap-learn` in the active Python environment:
///   pip install umap-learn
fn umap_via_python(
    data_buf: Vec<f32>,
    n: usize,
    k: usize,
    n_neighbors: usize,
    min_dist: f32,
) -> Result<Vec<f32>> {
    // pyo3 panics (rather than returning Err) if the Python interpreter is
    // absent.  Catch that so the caller gets a clean error instead of a crash.
    std::panic::catch_unwind(|| {
        Python::with_gil(|py| -> Result<Vec<f32>> {
        let locals = PyDict::new_bound(py);
        locals.set_item("data_flat",   data_buf)?;
        locals.set_item("n_cells",     n)?;
        locals.set_item("n_comp",      k)?;
        locals.set_item("n_neighbors", n_neighbors)?;
        locals.set_item("min_dist",    min_dist as f64)?;

        py.run_bound(
            r#"
import numpy as np
import umap as _umap

data = np.array(data_flat, dtype=np.float32).reshape(n_cells, n_comp)
reducer = _umap.UMAP(
    n_neighbors=n_neighbors,
    min_dist=min_dist,
    n_components=2,
    init='spectral',
)
embedding = reducer.fit_transform(data).astype(np.float32).flatten().tolist()
"#,
            None,
            Some(&locals),
        )
        .context("umap-learn failed (is umap-learn installed? pip install umap-learn)")?;

        let result: Vec<f32> = locals
            .get_item("embedding")?
            .context("embedding not set")?
            .extract()?;
        Ok(result)
    })
    })
    .unwrap_or_else(|_| {
        Err(anyhow::anyhow!(
            "Python interpreter not found or failed to initialise.\n\
             Make sure Python is available and umap-learn is installed:\n\
             \n  pip install umap-learn\n"
        ))
    })
}

// ── Public entry point ────────────────────────────────────────────────────────

/// Embed `embeddings` (n_cells × k_components) with umap-learn's spectral UMAP
/// and save an 800×600 PNG coloured by each cell's dominant sparse-PCA component.
///
/// `noise_frac` — cells whose max |score| < `noise_frac × global_max` are grey.
pub fn run_umap_and_plot(
    embeddings: &Mat<f64>,
    output_path: &str,
    noise_frac: f64,
    n_neighbors: usize,
    min_dist: f32,
) -> Result<()> {
    let n = embeddings.nrows();
    let k = embeddings.ncols();
    anyhow::ensure!(n > n_neighbors, "need more cells than n_neighbors ({n_neighbors})");

    // ── 1. Dominant-component label per cell ──────────────────────────────
    let global_max = (0..n).flat_map(|i| (0..k).map(move |c| embeddings.read(i,c).abs()))
        .fold(0.0_f64, f64::max);
    let labels = dominant_components(embeddings, global_max * noise_frac.clamp(0.0, 1.0));

    // ── 2. Convert to row-major f32 buffer ────────────────────────────────
    let buf = faer_to_vec_f32(embeddings);

    // ── 3. UMAP via Python (spectral init, same as scanpy) ────────────────
    eprintln!("[umap] calling umap-learn (n={n}, d={k}, n_neighbors={n_neighbors}) …");
    let emb_flat = umap_via_python(buf, n, k, n_neighbors, min_dist)?;
    // Layout: [x₀, y₀,  x₁, y₁, …]

    // ── 4. Axis bounds with 5 % padding ───────────────────────────────────
    let (mut x_min, mut x_max) = (f32::INFINITY, f32::NEG_INFINITY);
    let (mut y_min, mut y_max) = (f32::INFINITY, f32::NEG_INFINITY);
    for chunk in emb_flat.chunks_exact(2) {
        if chunk[0].is_finite() { x_min = x_min.min(chunk[0]); x_max = x_max.max(chunk[0]); }
        if chunk[1].is_finite() { y_min = y_min.min(chunk[1]); y_max = y_max.max(chunk[1]); }
    }
    let (x0, x1) = (x_min - (x_max-x_min)*0.05, x_max + (x_max-x_min)*0.05);
    let (y0, y1) = (y_min - (y_max-y_min)*0.05, y_max + (y_max-y_min)*0.05);

    // ── 5. Plot ───────────────────────────────────────────────────────────
    let root = BitMapBackend::new(output_path, (800, 600)).into_drawing_area();
    root.fill(&WHITE).context("fill background")?;

    let mut chart = ChartBuilder::on(&root)
        .margin(20)
        .x_label_area_size(0)
        .y_label_area_size(0)
        .build_cartesian_2d(x0..x1, y0..y1)
        .context("build chart")?;

    chart.configure_mesh().disable_x_axis().disable_y_axis().draw().context("draw grid")?;

    chart.draw_series(
        emb_flat.chunks_exact(2).enumerate().filter_map(|(i, chunk)| {
            let (x, y) = (chunk[0], chunk[1]);
            if !x.is_finite() || !y.is_finite() { return None; }
            Some(Circle::new((x, y), 2, component_color(labels[i], k).mix(0.7).filled()))
        })
    ).context("draw points")?;

    root.present().context("write PNG")?;
    eprintln!("[umap] saved → {output_path}");
    Ok(())
}
