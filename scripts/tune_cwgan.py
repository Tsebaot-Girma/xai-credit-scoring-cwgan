# scripts/tune_cwgan.py
"""
Optuna hyperparameter tuning for cWGAN-GP on the German Credit dataset.

Objective (composite score, higher = better):
  - GP convergence          (low, stable GP value)
  - Wasserstein stability   (low variance over last N epochs)
  - Generator loss stability (low variance = not mode-collapsing)
  - Wasserstein level       (negative = critic winning, which is normal)

Quick-start
-----------
1. DELETE any old DB before a fresh run:
       import os; os.remove("../models/cwgan_tuning/optuna_cwgan.db")
   Or set  RESUME = False  below.

2. Run from notebook:
       %run ../scripts/tune_cwgan.py
   Or call the helpers directly — see bottom of file.
"""

import os, sys, json, warnings
import numpy as np
import pandas as pd
import tensorflow as tf
import joblib
import optuna
from optuna.samplers import TPESampler
from optuna.pruners  import MedianPruner

warnings.filterwarnings("ignore")
optuna.logging.set_verbosity(optuna.logging.WARNING)

# ── resolve script dir so imports work whether run as script or via %run ──────
_file = globals().get("__file__", None)
SCRIPT_DIR = os.path.dirname(os.path.abspath(_file)) if _file else os.path.abspath("../scripts")
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

from cwgan_gp      import CWGANGP
from data_balancing import prepare_full_data_for_gan

# ══════════════════════════════════════════════════════════════════════════════
# CONFIG  ── adjust paths / budget to match your project layout
# ══════════════════════════════════════════════════════════════════════════════
DATA_PATH         = "../data/processed/german_processed_onehot.csv"
FEATURE_INFO_PATH = "../models/feature_info.pkl"
SAVE_DIR          = "../models/cwgan_tuning"
STUDY_NAME        = "cwgan_gp_tuning"
STUDY_DB          = f"sqlite:///{SAVE_DIR}/optuna_cwgan.db"

TARGET_COL        = "Class"
MINORITY_LABEL    = 1

# Tuning budget
N_TRIALS          = 40      # increase to 80-100 for thorough search
N_EPOCHS_TRIAL    = 300     # short run; enough to see convergence signal
BATCH_SIZE        = 64
N_CRITIC          = 3       # keep low during tuning to save time
STABILITY_WINDOW  = 50      # last N epochs used for stability metrics

# Set RESUME=False to wipe the DB and start fresh
RESUME            = False   # ← change to True once you have a clean DB

# Fixed architecture (not tuned here)
GEN_HIDDEN    = [256, 256]
CRITIC_HIDDEN = [256, 256]

SEED = 42
tf.random.set_seed(SEED)
np.random.seed(SEED)

# ══════════════════════════════════════════════════════════════════════════════
# Load data once (shared across all trials)
# ══════════════════════════════════════════════════════════════════════════════
print("Loading dataset …")
df           = pd.read_csv(DATA_PATH)
feature_info = joblib.load(FEATURE_INFO_PATH)
print(f"  Dataset shape : {df.shape}")
print(f"  Class distribution:\n{df[TARGET_COL].value_counts().to_string()}\n")


# ══════════════════════════════════════════════════════════════════════════════
# Composite stability / quality score
# ══════════════════════════════════════════════════════════════════════════════
def compute_trial_score(g_losses, w_distances, gp_vals, window=STABILITY_WINDOW):
    """
    Higher score = more stable, better-converging training.

    Components
    ----------
    gp_score      (0-1)  GP ≈ 0 is ideal; GP > 5 means critic instability
    w_stability   (0-1)  low CV of Wasserstein distance tail
    g_stability   (0-1)  low CV of generator loss tail
    w_level_score (0-1)  negative W-dist means real > fake (critic doing its job)
    """
    n = len(g_losses)
    if n < max(10, window // 2):
        return -999.0           # too few epochs — trial was pruned very early

    win = min(window, n)
    g_tail  = np.array(g_losses[-win:])
    w_tail  = np.array(w_distances[-win:])
    gp_tail = np.array(gp_vals[-win:])

    # 1. GP convergence
    gp_mean  = np.mean(gp_tail)
    gp_std   = np.std(gp_tail)
    gp_score = max(0.0, 1.0 - gp_mean / 5.0) - 0.1 * gp_std

    # 2. Wasserstein distance stability (coefficient of variation)
    w_mean      = np.mean(w_tail)
    w_cv        = np.std(w_tail) / (abs(w_mean) + 1e-8)
    w_stability = max(0.0, 1.0 - w_cv)

    # 3. Generator loss stability
    g_mean      = np.mean(g_tail)
    g_cv        = np.std(g_tail) / (abs(g_mean) + 1e-8)
    g_stability = max(0.0, 1.0 - g_cv)

    # 4. W-dist level: reward negative (real > fake), penalise extreme
    w_level_score = np.clip(-w_mean / 2.0, 0.0, 1.0)

    score = (
        0.35 * gp_score
      + 0.25 * w_stability
      + 0.20 * g_stability
      + 0.20 * w_level_score
    )
    return float(score)


# ══════════════════════════════════════════════════════════════════════════════
# Optuna objective
# ══════════════════════════════════════════════════════════════════════════════
def objective(trial: optuna.Trial) -> float:

    # ── Hyperparameter search space ───────────────────────────────────────────
    latent_dim         = trial.suggest_categorical("latent_dim",         [128, 256, 512])
    gp_weight          = trial.suggest_float(      "gp_weight",           1.0,  20.0)
    aux_weight         = trial.suggest_float(      "aux_weight",           0.1,   5.0)
    gen_lr             = trial.suggest_float(      "gen_lr",               5e-5,  5e-4, log=True)
    # critic LR expressed as a ratio of gen_lr (encodes WGAN-GP prior: critic ≤ gen LR)
    critic_lr_ratio    = trial.suggest_float(      "critic_lr_ratio",      0.1,   1.0)
    critic_lr          = gen_lr * critic_lr_ratio
    gumbel_temperature = trial.suggest_float(      "gumbel_temperature",   0.3,   1.0)
    use_cross_layers   = trial.suggest_categorical("use_cross_layers",    [True, False])
    use_lr_scheduling  = trial.suggest_categorical("use_lr_scheduling",   [True, False])

    # ── Attach derived critic_lr so it's visible in the DB ───────────────────
    trial.set_user_attr("critic_lr", round(critic_lr, 8))

    # ── Build dataset (re-shuffle per trial) ─────────────────────────────────
    dataset, _, n_num, n_cat_dims, _, _ = prepare_full_data_for_gan(
        df, feature_info, TARGET_COL, MINORITY_LABEL, BATCH_SIZE
    )

    # ── Build model ───────────────────────────────────────────────────────────
    # Use EXACTLY the parameter names from CWGANGP.__init__
    gan = CWGANGP(
        n_num=n_num,
        n_cat_dims=n_cat_dims,
        latent_dim=latent_dim,
        cond_dim=1,
        gen_hidden=GEN_HIDDEN,
        critic_hidden=CRITIC_HIDDEN,
        gp_weight=gp_weight,
        aux_weight=aux_weight,
        learning_rate=gen_lr,            # fallback if split LRs are None
        critic_learning_rate=critic_lr,  # matches __init__ exactly
        gen_learning_rate=gen_lr,        # matches __init__ exactly
        gumbel_temperature=gumbel_temperature,
        use_cross_layers=use_cross_layers,
        use_lr_scheduling=use_lr_scheduling,
    )

    # ── Short training (early stopping disabled — we score the trajectory) ────
    gan.train(
        dataset,
        epochs=N_EPOCHS_TRIAL,
        n_critic=N_CRITIC,
        verbose=False,
        early_stopping_patience=N_EPOCHS_TRIAL,   # effectively disabled
        restore_best=False,
    )

    # ── Mid-point pruning check ───────────────────────────────────────────────
    mid      = max(10, N_EPOCHS_TRIAL // 3)
    mid_win  = min(STABILITY_WINDOW, mid)
    mid_score = compute_trial_score(
        gan.g_losses[:mid], gan.w_distances[:mid], gan.gp_vals[:mid],
        window=mid_win,
    )
    trial.report(mid_score, step=mid)
    if trial.should_prune():
        del gan
        tf.keras.backend.clear_session()
        raise optuna.exceptions.TrialPruned()

    # ── Final score ───────────────────────────────────────────────────────────
    score = compute_trial_score(gan.g_losses, gan.w_distances, gan.gp_vals)

    # ── Checkpoint promising trials ───────────────────────────────────────────
    if score > 0.5:
        trial_dir = os.path.join(SAVE_DIR, f"trial_{trial.number}")
        try:
            gan.save(trial_dir)
            trial.set_user_attr("saved_path", trial_dir)
        except Exception as e:
            print(f"  [warn] Could not save trial {trial.number}: {e}")

    # ── Log diagnostics ───────────────────────────────────────────────────────
    tail = min(STABILITY_WINDOW, len(gan.g_losses))
    trial.set_user_attr("final_g_loss",  round(float(gan.g_losses[-1]),           4))
    trial.set_user_attr("final_w_dist",  round(float(gan.w_distances[-1]),        4))
    trial.set_user_attr("final_gp",      round(float(gan.gp_vals[-1]),            4))
    trial.set_user_attr("gp_mean_tail",  round(float(np.mean(gan.gp_vals[-tail:])), 4))

    del gan
    tf.keras.backend.clear_session()

    return score


# ══════════════════════════════════════════════════════════════════════════════
# Run study
# ══════════════════════════════════════════════════════════════════════════════
def run_tuning(n_trials=N_TRIALS, resume=RESUME):
    """
    Launch (or resume) the Optuna study.

    Parameters
    ----------
    n_trials : int   Number of trials to run.
    resume   : bool  True → load existing DB; False → delete DB and start fresh.
    """
    os.makedirs(SAVE_DIR, exist_ok=True)

    db_path = STUDY_DB.replace("sqlite:///", "")   # bare filesystem path

    if not resume and os.path.exists(db_path):
        os.remove(db_path)
        print(f"  Removed old DB: {db_path}")

    sampler = TPESampler(seed=SEED, n_startup_trials=max(5, n_trials // 8))
    pruner  = MedianPruner(n_startup_trials=5, n_warmup_steps=10)

    study = optuna.create_study(
        study_name=STUDY_NAME,
        direction="maximize",
        sampler=sampler,
        pruner=pruner,
        storage=STUDY_DB,
        load_if_exists=resume,      # False → always creates a fresh study
    )

    print("=" * 65)
    print(f"  Optuna cWGAN-GP Tuning  |  study: '{STUDY_NAME}'")
    print(f"  Trials: {n_trials}  |  Epochs/trial: {N_EPOCHS_TRIAL}")
    print(f"  Resume: {resume}  |  DB: {STUDY_DB}")
    print("=" * 65)

    # catch=(Exception,) means a single bad trial is logged but doesn't abort the study
    study.optimize(
        objective,
        n_trials=n_trials,
        show_progress_bar=True,
        gc_after_trial=True,
        catch=(Exception,),
    )

    return study


# ══════════════════════════════════════════════════════════════════════════════
# Results helpers
# ══════════════════════════════════════════════════════════════════════════════
def print_results(study):
    completed = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    pruned    = [t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED]
    failed    = [t for t in study.trials if t.state == optuna.trial.TrialState.FAIL]

    print("\n" + "=" * 65)
    print("  TUNING RESULTS")
    print("=" * 65)
    print(f"  Completed : {len(completed)}")
    print(f"  Pruned    : {len(pruned)}")
    print(f"  Failed    : {len(failed)}")

    if not completed:
        print("\n  No completed trials yet.")
        return None

    best = study.best_trial
    gen_lr    = best.params["gen_lr"]
    critic_lr = gen_lr * best.params["critic_lr_ratio"]

    print(f"\n  Best trial : #{best.number}  (score={best.value:.4f})")
    print("\n  Best hyperparameters:")
    for k, v in best.params.items():
        print(f"    {k:<25s}: {v}")
    print(f"    {'gen_lr (effective)':<25s}: {gen_lr:.2e}")
    print(f"    {'critic_lr (derived)':<25s}: {critic_lr:.2e}")

    print("\n  Diagnostics at end of best trial:")
    for k, v in best.user_attrs.items():
        print(f"    {k:<25s}: {v}")

    top5 = sorted(completed, key=lambda t: t.value, reverse=True)[:5]
    header = (f"  {'#':>4}  {'Score':>7}  {'latent':>7}  {'gp_w':>6}  "
              f"{'aux_w':>6}  {'gen_lr':>9}  {'cr_ratio':>8}  "
              f"{'gumbel':>7}  {'cross':>6}  {'lr_sch':>6}")
    print(f"\n  Top 5 trials:\n{header}")
    print("  " + "-" * (len(header) - 2))
    for t in top5:
        p = t.params
        print(
            f"  {t.number:>4}  {t.value:>7.4f}"
            f"  {p['latent_dim']:>7}"
            f"  {p['gp_weight']:>6.2f}"
            f"  {p['aux_weight']:>6.2f}"
            f"  {p['gen_lr']:>9.2e}"
            f"  {p['critic_lr_ratio']:>8.3f}"
            f"  {p['gumbel_temperature']:>7.3f}"
            f"  {str(p['use_cross_layers']):>6}"
            f"  {str(p['use_lr_scheduling']):>6}"
        )
    return best


def save_best_config(study, out_path=None):
    """Persist best hyperparameters + derived values to JSON."""
    if out_path is None:
        out_path = os.path.join(SAVE_DIR, "best_config.json")

    best      = study.best_trial
    gen_lr    = best.params["gen_lr"]
    critic_lr = gen_lr * best.params["critic_lr_ratio"]

    config = {
        "trial_number":       best.number,
        "score":              round(best.value, 4),
        "latent_dim":         best.params["latent_dim"],
        "gp_weight":          round(best.params["gp_weight"],          4),
        "aux_weight":         round(best.params["aux_weight"],          4),
        "gen_lr":             gen_lr,
        "critic_lr":          critic_lr,
        "critic_lr_ratio":    round(best.params["critic_lr_ratio"],     4),
        "gumbel_temperature": round(best.params["gumbel_temperature"],  4),
        "use_cross_layers":   best.params["use_cross_layers"],
        "use_lr_scheduling":  best.params["use_lr_scheduling"],
        "gen_hidden":         GEN_HIDDEN,
        "critic_hidden":      CRITIC_HIDDEN,
        "diagnostics":        best.user_attrs,
    }

    with open(out_path, "w") as f:
        json.dump(config, f, indent=2)
    print(f"\n  Best config saved → {out_path}")
    return config


def retrain_best(config, epochs_full=6000, early_stopping_patience=2000):
    """Retrain with the best config for the full epoch budget."""
    from data_balancing import train_cwgan_full

    print("\n" + "=" * 65)
    print("  RETRAINING WITH BEST CONFIG")
    print("=" * 65)

    df_local = pd.read_csv(DATA_PATH)
    fi       = joblib.load(FEATURE_INFO_PATH)

    gan, scaler = train_cwgan_full(
        df_train=df_local,
        feature_info=fi,
        target_col=TARGET_COL,
        minority_label=MINORITY_LABEL,
        latent_dim=config["latent_dim"],
        epochs=epochs_full,
        batch_size=BATCH_SIZE,
        n_critic=N_CRITIC,
        gp_weight=config["gp_weight"],
        aux_weight=config["aux_weight"],
        learning_rate=config["gen_lr"],
        critic_learning_rate=config["critic_lr"],
        gen_learning_rate=config["gen_lr"],
        gumbel_temperature=config["gumbel_temperature"],
        use_cross_layers=config["use_cross_layers"],
        use_lr_scheduling=config["use_lr_scheduling"],
        early_stopping_patience=early_stopping_patience,
        verbose=True,
        save_path=os.path.join(SAVE_DIR, "best_retrained"),
        restore_best=True,
    )
    return gan, scaler


def plot_importances(study, save_path=None):
    """Plot optimisation history and hyperparameter importances."""
    try:
        import matplotlib.pyplot as plt
        from optuna.visualization.matplotlib import (
            plot_param_importances,
            plot_optimization_history,
        )
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        plot_optimization_history(study, ax=axes[0])
        axes[0].set_title("Optimisation History")
        plot_param_importances(study, ax=axes[1])
        axes[1].set_title("Hyperparameter Importances")
        plt.tight_layout()
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
            print(f"  Saved → {save_path}")
        plt.show()
    except ImportError:
        print("  matplotlib / optuna.visualization not available.")


# ══════════════════════════════════════════════════════════════════════════════
# Entrypoint (direct script execution)
# ══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    study  = run_tuning(n_trials=N_TRIALS, resume=RESUME)
    best_t = print_results(study)
    if best_t is not None:
        config = save_best_config(study)
        plot_importances(
            study,
            save_path=os.path.join(SAVE_DIR, "optuna_importance.png"),
        )
        # Uncomment to immediately retrain with best config:
        # gan, scaler = retrain_best(config, epochs_full=6000)
