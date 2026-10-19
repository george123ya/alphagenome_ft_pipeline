"""AlphaGenome model helpers (need JAX and the patched alphagenome_ft)."""
import numpy as np


def head_specs(cfg):
    """Register the configured head and return (specs, head_id).

    organism=None is required. With organism="MUS_MUSCULUS" alphagenome_ft gives the head a single
    organism slot; mouse is organism index 1, which is out of range. JAX clamps the index in the
    forward pass (predictions look normal) but drops it in the backward pass, so every head gradient
    is exactly zero and nothing trains. organism=None builds metadata for both organisms (2 slots).
    """
    from alphagenome_ft.finetune import prepare_head_specs, register_predefined_heads
    specs = prepare_head_specs(cfg.targets_config(), organism=None)
    register_predefined_heads(specs)
    return specs, specs[0].head_id


def organism_settings(cfg):
    """Local files only: the configured FASTA for our organism, nothing is downloaded."""
    from alphagenome.models import dna_client
    from alphagenome_research.model import dna_model
    ours = getattr(dna_client.Organism, cfg.organism)
    return {o: dna_model.OrganismSettings(fasta_path=str(cfg.fasta)) if o == ours else dna_model.OrganismSettings()
            for o in (dna_client.Organism.HOMO_SAPIENS, dna_client.Organism.MUS_MUSCULUS)}


def load_finetuned(cfg, ckpt):
    import jax
    from alphagenome_ft import load_checkpoint
    return load_checkpoint(str(ckpt), base_model_version="all_folds", base_checkpoint_path=str(cfg.weights),
                           organism_settings=organism_settings(cfg), device=jax.devices()[0])


def best_checkpoint(cfg):
    for name in ("best", "last"):
        if (cfg.ckpt_dir / name).exists():
            return cfg.ckpt_dir / name
    raise FileNotFoundError(f"no checkpoint in {cfg.ckpt_dir}: run 02_train.py first")


def make_predict(model, cfg, head):
    """jitted f(one_hot (1, L, 4)) -> head output tree, forward strand, configured organism."""
    import jax
    import jax.numpy as jnp
    import alphagenome.models.dna_client as dna_client
    from alphagenome_research.model import dna_model as rdm
    org = getattr(dna_client.Organism, cfg.organism)
    org_idx = jnp.full((1,), rdm.convert_to_organism_index(org), jnp.int32)
    neg = jnp.zeros((1,), bool)
    sr = jax.device_put(model._metadata[org].strand_reindexing)  # a pytree, not an array
    f = jax.jit(lambda p, s, x: model._predict(p, s, x, org_idx, negative_strand_mask=neg,
                                               strand_reindexing=sr)[head])
    return lambda x: f(model._params, model._state, jnp.asarray(x))


def head_track(pred, length, ntracks):
    """(B, length, T) 1 bp array from a head output (128 bp output is spread evenly over its bin)."""
    import jax
    leaves = [np.asarray(x) for x in jax.tree_util.tree_leaves(pred)
              if hasattr(x, "shape") and x.ndim == 3 and x.shape[-1] == ntracks]
    for x in leaves:
        if x.shape[1] == length:
            return x
    for x in leaves:
        if x.shape[1] == length // 128:
            return np.repeat(x, 128, axis=1) / 128.0
    raise RuntimeError(f"no usable head output, shapes={[x.shape for x in leaves]}")


def gpu_memory_gib():
    import jax
    try:
        s = jax.devices()[0].memory_stats()
        return round(s.get("peak_bytes_in_use", 0) / 2**30, 2)
    except Exception:  # noqa: BLE001  (CPU has no memory stats)
        return None
