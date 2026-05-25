use anyhow::Result;
use clap::Parser;
use faer;
use std::fs::OpenOptions;
use std::io::Write;
use std::time::Instant;

use rmt_spca_io::{load_and_prepare, project_cells, run_umap_and_plot, Preprocessor};
use rmt_spca_io::reader::ScDataReader;
use rmt_spca::rmt::RmtTheory;
use rmt_spca::spca::{FistaConfig, SparsePCA};
use rmt_spca::markers::print_marker_table;

// ── CLI ───────────────────────────────────────────────────────────────────────

#[derive(Parser)]
#[command(about = "RMT-guided Sparse PCA validation for .h5ad files")]
struct Args {
    /// Path to the input .h5ad file.
    path: String,

    /// Use only the first N cells (rows).
    #[arg(long)]
    n_cells: Option<usize>,

    /// L1 sparsity penalty λ (absolute).
    #[arg(long, default_value_t = 0.1)]
    lambda: f64,

    /// Set λ = F × λ+, overriding --lambda.  ~0.3 targets 200–500 genes/component.
    #[arg(long)]
    lambda_frac: Option<f64>,

    /// Max Sinkhorn-Knopp iterations for biwhitening.
    #[arg(long, default_value_t = 1000)]
    bw_max_iter: usize,

    /// Sinkhorn under-relaxation α ∈ (0,1].  Try 0.5–0.8 if biwhitening oscillates.
    #[arg(long, default_value_t = 1.0)]
    bw_damp: f64,

    /// Top marker genes printed per component (0 = suppress).
    #[arg(long, default_value_t = 10)]
    top_markers: usize,

    /// Skip log-normalisation (use with already-normalised data).
    #[arg(long)]
    no_log: bool,

    /// Bypass the `is_likely_log_normalized` auto-detect and use the raw matrix
    /// as-is (no log-normalisation, no warning).  Implies `--no-log`.
    #[arg(long)]
    skip_preprocess: bool,

    /// Transform label recorded in telemetry (free-form, e.g. "log1p", "raw").
    #[arg(long, default_value = "auto")]
    transform: String,

    /// Replicate index recorded in telemetry.
    #[arg(long, default_value_t = 0)]
    rep: usize,

    /// Append one JSON line of run telemetry to this path.
    #[arg(long)]
    telemetry_out: Option<String>,

    /// Project cells onto components and save a UMAP scatter-plot.
    #[arg(long)]
    umap: bool,

    /// Output path for the UMAP PNG.
    #[arg(long, default_value = "umap.png")]
    umap_out: String,

    /// Cells whose max |component score| is below this fraction of the global
    /// maximum are drawn grey (unassigned).  Increase to reduce the background
    /// mass and make component clusters stand out; try 0.10–0.30.
    #[arg(long, default_value_t = 0.10)]
    umap_noise_frac: f64,

    /// UMAP n_neighbors (passed to umap-learn).
    #[arg(long, default_value_t = 15)]
    umap_n_neighbors: usize,

    /// UMAP min_dist (passed to umap-learn).
    #[arg(long, default_value_t = 0.1)]
    umap_min_dist: f32,
}

// ── Main ──────────────────────────────────────────────────────────────────────

fn main() -> Result<()> {
    let args = Args::parse();
    let force_log = !args.no_log && !args.skip_preprocess;
    let t_start = Instant::now();
    let num_cpus = std::thread::available_parallelism().map(|n| n.get()).unwrap_or(1);
    eprintln!("Using {num_cpus} CPU core(s) (std::thread::available_parallelism).");

    // -----------------------------------------------------------------------
    // 1. Load (+ auto-normalise unless --skip-preprocess)
    // -----------------------------------------------------------------------
    eprintln!("Loading {} …", args.path);
    let gene_names = ScDataReader::load_var_names(&args.path)?;
    let mut mat = if args.skip_preprocess {
        eprintln!("--skip-preprocess: bypassing is_likely_log_normalized auto-detect.");
        ScDataReader::load_h5ad_as_faer(&args.path)?
    } else {
        load_and_prepare(&args.path, force_log)?
    };

    if let Some(n) = args.n_cells {
        let n = n.min(mat.nrows());
        let p = mat.ncols();
        eprintln!("Subsampling to {n} cells (of {}) …", mat.nrows());
        mat = faer::Mat::from_fn(n, p, |i, j| mat.read(i, j));
    }
    let (n, p) = (mat.nrows(), mat.ncols());
    println!("Loaded matrix: {n} cells × {p} genes");
    println!("Log-normalised: {}", Preprocessor::is_likely_log_normalized(&mat));

    // -----------------------------------------------------------------------
    // 2. RMT threshold (pre-filter estimate)
    // -----------------------------------------------------------------------
    let rmt_pre = RmtTheory { q: p as f64 / n as f64 };
    println!(
        "MP bulk edge λ+ = {:.4}  (q = p/n = {:.4}, pre-filter)",
        rmt_pre.lambda_plus(),
        rmt_pre.q
    );

    // -----------------------------------------------------------------------
    // 3. Sparse PCA
    // -----------------------------------------------------------------------
    eprintln!("Running Sparse PCA (λ = {}) …", args.lambda);
    let config = FistaConfig {
        lambda: args.lambda,
        lambda_frac: args.lambda_frac,
        verbose: true,
        bw_max_iter: args.bw_max_iter,
        bw_damp: args.bw_damp,
        ..FistaConfig::default()
    };
    let result = SparsePCA::new(config).fit(&mat);
    let (out_p, k) = (result.components.nrows(), result.components.ncols());
    println!("Components found: {k}  (each length {out_p})");

    let nnz = (0..out_p)
        .flat_map(|i| (0..k).map(move |j| (i, j)))
        .filter(|&(i, j)| result.components.read(i, j).abs() > 1e-6)
        .count();
    println!(
        "Non-zero loadings: {nnz}/{} ({:.1}%)",
        out_p * k,
        100.0 * nnz as f64 / (out_p * k) as f64
    );

    let lp = result.lambda_plus;
    let rmt = RmtTheory { q: result.q };
    println!("MP bulk edge λ+ = {lp:.4}  (q = {:.4}, post-filter)", result.q);
    println!("Noise scale σ²  = {:.4}  (should be ≈1 for well-biwhitened data)",
             result.sigma_sq);

    // -----------------------------------------------------------------------
    // 4. KS validation against MP bulk
    // -----------------------------------------------------------------------
    match result.ks_distance {
        Some(ks) => {
            println!("KS distance (bulk eigenspectrum) from MP: {ks:.4}");
            if ks < 0.10 {
                println!("✓ Bulk eigenspectrum consistent with MP — biwhitening looks good.");
            } else {
                println!("⚠ KS = {ks:.4} — bulk eigenspectrum deviates from MP.");
            }
        }
        None => println!("KS diagnostic: unavailable (fast eigensolver or disabled)."),
    }

    // -----------------------------------------------------------------------
    // 5. Per-component RMT overlap predictions
    // -----------------------------------------------------------------------
    let overlaps: Vec<f64> = result.eigenvalues.iter()
        .map(|&ev| if ev > lp { rmt.predicted_overlap(ev) } else { f64::NAN })
        .collect();

    println!("\nComponent  λ_signal (RQ)  predicted_overlap");
    for c in 0..k {
        let lambda_est = result.eigenvalues[c];
        if lambda_est > lp {
            println!("  {c:>9}  {lambda_est:>12.4}  {:.4}", overlaps[c]);
        } else {
            println!("  {c:>9}  {lambda_est:>12.4}  (below bulk edge)");
        }
    }

    // -----------------------------------------------------------------------
    // 6. Marker genes per component
    // -----------------------------------------------------------------------
    if args.top_markers > 0 {
        let names: &[String] = &gene_names[..gene_names.len().min(out_p)];
        print_marker_table(
            &result.components,
            names,
            &result.eigenvalues,
            &overlaps,
            args.top_markers,
        );
    }

    // -----------------------------------------------------------------------
    // 7. UMAP  (--umap flag)
    // -----------------------------------------------------------------------
    if args.umap {
        // Project the n×p expression matrix onto the p×k loading matrix to
        // obtain n×k cell scores.  `project_cells` handles the case where
        // SparsePCA dropped all-zero genes (out_p ≤ p) by truncating mat.
        let scores = project_cells(&mat, &result.components);
        run_umap_and_plot(
            &scores, &args.umap_out,
            args.umap_noise_frac,
            args.umap_n_neighbors,
            args.umap_min_dist,
        )?;
    }

    // -----------------------------------------------------------------------
    // 8. Telemetry (--telemetry-out)
    // -----------------------------------------------------------------------
    if let Some(ref path) = args.telemetry_out {
        let runtime_s = t_start.elapsed().as_secs_f64();
        let record = serde_json::json!({
            "transform": args.transform,
            "N": n,
            "rep": args.rep,
            "lambda_plus": result.lambda_plus,
            "sigma_sq": result.sigma_sq,
            "ks_distance": result.ks_distance,
            "sk_iters": result.sk_iters,
            "sk_converged": result.sk_converged,
            "sk_residual": result.sk_residual,
            "s_eigenvalues": result.s_eigenvalues,
            "runtime_s": runtime_s,
            "num_cpus": num_cpus,
        });
        let mut f = OpenOptions::new().create(true).append(true).open(path)?;
        writeln!(f, "{}", record)?;
        eprintln!("Telemetry appended to {path}");
    }

    Ok(())
}
