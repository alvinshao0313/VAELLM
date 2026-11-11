"""Semantic random streams and two-level candidate/artifact cache identities."""
from .contracts import canonical_json, digest, require, validate_recipe, validate_run_identity

SEED_VERSION = 'semantic_uint32_model_category_stage_purpose_v0'


def semantic_seed(model_identity, semantic_category_id, stage, purpose, search_seed):
    require(isinstance(semantic_category_id, str) and semantic_category_id and isinstance(stage, str) and stage
        and isinstance(purpose, str) and purpose, 'Category, stage and purpose are semantic strings, never positions.')
    require(isinstance(search_seed, int) and not isinstance(search_seed, bool) and search_seed >= 0, 'Explicit nonnegative search_seed required.')
    value = digest(dict(version=SEED_VERSION, model=model_identity, category_id=semantic_category_id,
        stage=stage, purpose=purpose, search_seed=search_seed))
    # Both native set_seed and Transformers reach np.random.seed(uint32).
    return int(value[:8], 16)


def candidate_id(identity, recipe):
    identity, recipe = validate_run_identity(identity), validate_recipe(recipe)
    return digest(dict(schema='candidate_plan_v0', identity=identity, recipe=recipe,
        init_contract=dict(seed_version=SEED_VERSION, shape_pairing='Same semantic seed for paired same-shape candidates; no bitwise claim for changed structure')))


def recipe_id(recipe):
    return digest(validate_recipe(recipe))


def artifact_id(planned_candidate_id, runtime_resolution, checkpoint_digest):
    require(isinstance(planned_candidate_id, str) and planned_candidate_id and runtime_resolution
        and isinstance(checkpoint_digest, str) and checkpoint_digest, 'Fitted artifacts need planned identity, actual runtime evidence and payload digest.')
    return digest(dict(candidate_id=planned_candidate_id, runtime_resolution=runtime_resolution, checkpoint_digest=checkpoint_digest))


def cache_key(identity, recipe, purpose, token_order_digest):
    require(isinstance(purpose, str) and purpose and isinstance(token_order_digest, str) and token_order_digest,
        'Cache reuse must bind purpose and exact token/record order.')
    return digest(dict(candidate_id=candidate_id(identity, recipe), purpose=purpose, token_order_digest=token_order_digest))


def reusable_result(record, identity, recipe, purpose, token_order_digest):
    expected = candidate_id(identity, recipe)
    if record.get('candidate_id') != expected or record.get('cache_key') != cache_key(identity, recipe, purpose, token_order_digest):
        return False
    if record.get('status') != 'completed' or record.get('exit_code') != 0 or not record.get('runtime_resolution') or not record.get('checkpoint_digest'):
        return False
    resolution = record['runtime_resolution']
    if recipe['optimizer']['kind'] == 'vae_muon' and not resolution.get('kappa_owner_stage_evidence'):
        return False
    return record.get('artifact_id') == artifact_id(expected, resolution, record['checkpoint_digest'])
