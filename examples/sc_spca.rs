use anyhow::{Context, Result};
use clap::Parser;
use faer::{self, Col};
use std::fs::OpenOptions;
use std::io::Write;
use std::time::Instant;

use rmt_spca_io::{load_and_prepare, project_cells, run_umap_and_plot, Preprocessor};
use rmt_spca_io::reader::{CsrRowReader, ScDataReader};
use rmt_spca::biwhitening::Biwhitener;
use rmt_spca::sparse::CsrMatrix;
use rmt_spca::streaming::{
    cell_factors, gene_nnz, reset_rare_genes, subsample_spectrum, CovAccumulator, GeneMoments,
};
use rmt_spca::rmt::RmtTheory;
use rmt_spca::spca::{EigensolverMode, FistaConfig, SparsePCA};
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

    /// Set λ = F × λ+, overriding --lambda. Has no relation to the spectrum:
    /// on A549 0.3 collapsed 50/53 components to single genes. Prefer --auto-penalty.
    #[arg(long)]
    lambda_frac: Option<f64>,

    /// Choose λ by the paper's Eq. 8 (root-find λ* so that the sparse subspace
    /// sits at the RMT-predicted distance from the outlier eigenspace), then
    /// fit at SCALE × λ* (reference default 0.6). Overrides --lambda/--lambda-frac.
    /// Costs ~10 FISTA runs.
    #[arg(long, num_args = 0..=1, default_missing_value = "0.6", value_name = "SCALE")]
    auto_penalty: Option<f64>,

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

    /// **P2.A** — load precomputed biwhitening factors from a JSON file
    /// containing `{"c": [...], "d": [...]}` (lengths n and p respectively,
    /// matching the input matrix after Stage 0 zero-row/col filtering).
    /// When set, Sinkhorn-Knopp biwhitening is skipped and these factors are
    /// applied directly. Enables fixed-biwhitening tests across N-subsamples
    /// of the same population.
    #[arg(long)]
    bw_factors_in: Option<String>,

    /// **P2.B** — write biwhitening factors, unrescaled eigenvalues, and
    /// top-K eigenvectors of the σ̂²-rescaled covariance to this path as one
    /// JSON object. Use the companion `--top-eigvec-count` to control K.
    #[arg(long)]
    bw_state_out: Option<String>,

    /// **P2.B** — how many top eigenvectors to store/dump (default 20).
    #[arg(long, default_value_t = 20)]
    top_eigvec_count: usize,

    /// Write sparse loadings (p×k, row-major), biwhitening factors and
    /// eigenvalues to this JSON path, for downstream clustering in Python.
    #[arg(long)]
    components_out: Option<String>,

    /// Keep X sparse end to end (SparsePCA::fit_csr): memory O(nnz + p²)
    /// instead of several dense n×p copies. Requires --skip-preprocess and CSR X;
    /// not compatible with --umap or --n-cells.
    #[arg(long)]
    sparse: bool,

    /// σ² and KS from the spectrum of N evenly spaced cells (m×m Gram) instead
    /// of the full p×p EVD (EigensolverMode::Subsample). 0 = full EVD (default).
    #[arg(long, default_value_t = 0)]
    eig_subsample: usize,

    /// Ceiling on the signal components searched for (FistaConfig::k_max).
    #[arg(long, default_value_t = 20)]
    k_max: usize,

    /// One S-product per FISTA iteration instead of two (FistaConfig::fista_reuse_products).
    #[arg(long)]
    fista_reuse: bool,

    /// Stream X from disk in blocks of this many cells (memory O(p² + block)):
    /// gene factors from Sinkhorn on --bw-subsample cells, refined over all cells
    /// --refine-passes times, then one pass accumulates the covariance.
    /// Requires --skip-preprocess and CSR X without empty cells/genes. 0 = off.
    #[arg(long, default_value_t = 0)]
    stream_block: usize,

    /// Cells (evenly spaced) for the Sinkhorn warm start of --stream-block.
    #[arg(long, default_value_t = 50_000)]
    bw_subsample: usize,

    /// Full-data gene-factor refinement passes for --stream-block. Rare genes
    /// converge at ≈ √(n/(k·p)) per pass; 4 brings them within ~7%.
    #[arg(long, default_value_t = 4)]
    refine_passes: usize,

    /// --stream-block: genes seen in fewer subsample cells than this get their
    /// Sinkhorn factor reset before refinement (it can drift to ~1e11 and erase
    /// the cells expressing them; see rmt_spca::streaming::reset_rare_genes).
    #[arg(long, default_value_t = 5)]
    min_gene_sub_cells: usize,
}

/// Evenly spaced indices t·n/m, t < m (stratified on files sorted by sample).
fn spaced(n: usize, m: usize) -> Vec<usize> {
    let m = m.clamp(1, n);
    (0..m).map(|t| t * n / m).collect()
}

/// The streaming pipeline (see `rmt_spca::streaming`), reading row blocks from disk.
fn fit_streaming(r: &CsrRowReader, args: &Args, config: FistaConfig) -> Result<rmt_spca::spca::SparsePCAResult> {
    let (n, p, bs) = (r.n, r.p, args.stream_block);
    let blocks = || (0..n).step_by(bs).map(move |a| (a, (a + bs).min(n)));
    let no_empty_rows = |blk: &CsrMatrix, a: usize| -> Result<()> {
        if let Some(i) = (0..blk.nrows()).find(|&i| blk.row(i).all(|(_, v)| v == 0.0)) {
            anyhow::bail!("cell {} is empty: prefilter empty cells before --stream-block", a + i);
        }
        Ok(())
    };

    let t = Instant::now();
    let sub = r.rows(&spaced(n, args.bw_subsample))?;
    no_empty_rows(&sub, 0)?;
    let (_, mut d, iters, ok, res) = Biwhitener {
        max_iter: args.bw_max_iter,
        damp: args.bw_damp,
        ..Biwhitener::default()
    }
    .compute_csr(&sub);
    eprintln!(
        "[stream] Sinkhorn on {} cells: {iters} iters, converged={ok}, residual={res:.2e} ({:.1}s)",
        sub.nrows(),
        t.elapsed().as_secs_f64()
    );
    let reset = reset_rare_genes(&mut d, &gene_nnz(&sub), args.min_gene_sub_cells);
    if reset > 0 && args.refine_passes == 0 {
        anyhow::bail!("{reset} genes seen in < {} subsample cells need --refine-passes ≥ 1", args.min_gene_sub_cells);
    }
    eprintln!("[stream] reset {reset} rare-gene factors (< {} subsample cells)", args.min_gene_sub_cells);

    for pass in 0..args.refine_passes {
        let t = Instant::now();
        let mut gm = GeneMoments::new(p);
        for (a, b) in blocks() {
            let blk = r.range(a, b)?;
            no_empty_rows(&blk, a)?;
            gm.add(&blk, &cell_factors(&blk, &d));
        }
        d = gm.finish();
        eprintln!("[stream] refinement pass {} ({:.1}s)", pass + 1, t.elapsed().as_secs_f64());
    }

    let t = Instant::now();
    let mut acc = CovAccumulator::new(p);
    let mut c_all = Vec::with_capacity(n);
    let mut gene_seen = vec![false; p];
    for (a, b) in blocks() {
        let blk = r.range(a, b)?;
        no_empty_rows(&blk, a)?;
        for i in 0..blk.nrows() {
            for (j, _) in blk.row(i) {
                gene_seen[j] = true;
            }
        }
        let c = cell_factors(&blk, &d);
        c_all.extend((0..c.nrows()).map(|i| c.read(i)));
        acc.add(&blk, &c, &d);
    }
    if let Some(j) = gene_seen.iter().position(|&s| !s) {
        anyhow::bail!("gene {j} is never expressed: prefilter empty genes before --stream-block");
    }
    let (s, mu) = acc.finish();
    eprintln!("[stream] covariance pass ({:.1}s)", t.elapsed().as_secs_f64());

    let spectrum = match config.eigensolver {
        EigensolverMode::Subsample { n_cells } => {
            let idx = spaced(n, n_cells);
            let blk = r.rows(&idx)?;
            let c_s = Col::from_fn(idx.len(), |t| c_all[idx[t]]);
            let all: Vec<usize> = (0..idx.len()).collect();
            Some(subsample_spectrum(&blk.centred_rows(&all, &c_s, &d, &mu)))
        }
        _ => None,
    };
    let c = Col::from_fn(n, |i| c_all[i]);
    Ok(SparsePCA::new(config).fit_covariance(s, n, c, d, spectrum))
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
    let stream = args.stream_block > 0;
    if (args.sparse || stream) && (!args.skip_preprocess || args.umap || args.n_cells.is_some()) {
        anyhow::bail!("--sparse / --stream-block require --skip-preprocess and exclude --umap / --n-cells");
    }
    if stream && args.bw_factors_in.is_some() {
        anyhow::bail!("--stream-block computes its own factors; drop --bw-factors-in");
    }
    let reader = if stream { Some(CsrRowReader::open(&args.path)?) } else { None };
    let csr = if args.sparse && !stream {
        Some(ScDataReader::load_h5ad_as_csr(&args.path)?)
    } else {
        None
    };
    let mut mat = if args.sparse || stream {
        faer::Mat::zeros(0, 0)
    } else if args.skip_preprocess {
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
    let (n, p) = match (&reader, &csr) {
        (Some(r), _) => (r.n, r.p),
        (_, Some(x)) => (x.nrows(), x.ncols()),
        _ => (mat.nrows(), mat.ncols()),
    };
    println!("Loaded matrix: {n} cells × {p} genes");
    if csr.is_none() && reader.is_none() {
        println!("Log-normalised: {}", Preprocessor::is_likely_log_normalized(&mat));
    }

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
    match (args.auto_penalty, args.lambda_frac) {
        (Some(sc), _) => eprintln!("Running Sparse PCA (λ by Eq. 8, scale {sc}) …"),
        (None, Some(f)) => eprintln!("Running Sparse PCA (λ = {f} × λ+) …"),
        (None, None) => eprintln!("Running Sparse PCA (λ = {}) …", args.lambda),
    }

    // P2.A: load precomputed biwhitening factors if provided.
    let precomputed_factors = match args.bw_factors_in.as_deref() {
        Some(path) => {
            let txt = std::fs::read_to_string(path)
                .with_context(|| format!("reading bw_factors_in from {path}"))?;
            let v: serde_json::Value = serde_json::from_str(&txt)
                .with_context(|| format!("parsing {path} as JSON"))?;
            let c_arr = v.get("c").and_then(|x| x.as_array())
                .with_context(|| format!("{path} missing 'c' array"))?;
            let d_arr = v.get("d").and_then(|x| x.as_array())
                .with_context(|| format!("{path} missing 'd' array"))?;
            let c_vec: Vec<f64> = c_arr.iter().map(|x| x.as_f64().unwrap_or(0.0)).collect();
            let d_vec: Vec<f64> = d_arr.iter().map(|x| x.as_f64().unwrap_or(0.0)).collect();
            eprintln!("[P2.A] loaded precomputed biwhitening factors from {path} (|c|={}, |d|={})",
                c_vec.len(), d_vec.len());
            let c = Col::from_fn(c_vec.len(), |i| c_vec[i]);
            let d = Col::from_fn(d_vec.len(), |j| d_vec[j]);
            Some((c, d))
        }
        None => None,
    };

    let config = FistaConfig {
        lambda: args.lambda,
        lambda_frac: args.lambda_frac,
        auto_penalty: args.auto_penalty,
        verbose: true,
        bw_max_iter: args.bw_max_iter,
        bw_damp: args.bw_damp,
        precomputed_factors,
        top_eigvec_count: args.top_eigvec_count,
        eigensolver: if args.eig_subsample > 0 {
            EigensolverMode::Subsample { n_cells: args.eig_subsample }
        } else {
            EigensolverMode::Full
        },
        k_max: args.k_max,
        fista_reuse_products: args.fista_reuse,
        ..FistaConfig::default()
    };
    let result = match (&reader, &csr) {
        (Some(r), _) => fit_streaming(r, &args, config)?,
        (_, Some(x)) => SparsePCA::new(config).fit_csr(x),
        _ => SparsePCA::new(config).fit(&mat),
    };
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
    println!(
        "L1 penalty λ    = {:.4e}{}  subspace distance Σ(1−σ²) = {:.3} vs Eq. 8 target {:.3}",
        result.lambda,
        match result.lambda_star {
            Some(st) => format!("  (Eq. 8 root λ* = {st:.4e})"),
            None => String::new(),
        },
        result.subspace_distance,
        result.target_distance
    );
    let singletons = (0..k)
        .filter(|&j| (0..out_p).map(|i| result.components.read(i, j).abs()).fold(0.0, f64::max) > 0.9)
        .count();
    if singletons > 0 {
        println!("⚠ {singletons}/{k} components are a single gene (max|w| > 0.9): over-penalised. Use --auto-penalty.");
    }

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
    // P2.B: dump biwhitening state (factors, unrescaled eigenvalues, top eigenvectors)
    if let Some(ref path) = args.bw_state_out {
        let dump = serde_json::json!({
            "bw_c": result.bw_c,
            "bw_d": result.bw_d,
            "s_eigenvalues_unrescaled": result.s_eigenvalues_unrescaled,
            "top_eigenvectors": result.top_eigenvectors,
            "sigma_sq": result.sigma_sq,
            "q": result.q,
            "lambda_plus": result.lambda_plus,
            "n_cells": n,
            "n_genes": p,
        });
        std::fs::write(path, serde_json::to_string(&dump)?)?;
        eprintln!("[P2.B] biwhitening state dumped to {path}");
    }

    if let Some(ref path) = args.components_out {
        let w = &result.components;
        let rows: Vec<Vec<f64>> = (0..w.nrows())
            .map(|i| (0..w.ncols()).map(|c| w.read(i, c)).collect())
            .collect();
        let dump = serde_json::json!({
            "components": rows,
            "eigenvalues": result.eigenvalues,
            "bw_c": result.bw_c,
            "bw_d": result.bw_d,
            "sigma_sq": result.sigma_sq,
            "lambda_plus": result.lambda_plus,
            "lambda": result.lambda,
            "lambda_star": result.lambda_star,
            "subspace_distance": result.subspace_distance,
            "target_distance": result.target_distance,
            "ks_distance": result.ks_distance,
            "collapsed_cells": result.collapsed_cells,
            "max_gene_factor_ratio": result.max_gene_factor_ratio,
        });
        std::fs::write(path, serde_json::to_string(&dump)?)?;
        eprintln!("components dumped to {path}");
    }

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
