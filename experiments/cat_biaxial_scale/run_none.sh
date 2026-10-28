#!/usr/bin/env bash
set -euo pipefail
export OMP_NUM_THREADS=4
export CUBLAS_WORKSPACE_CONFIG=:4096:8
input=/home/star/VAELLM/result/qwen_cat_biaxial_20261022/prepared/inputs.pt
output=/home/star/VAELLM/result/qwen_cat_biaxial_20261022/formal
mkdir -p "$output"

python -m experiments.cat_biaxial_scale.run --input-file "$input" --output-dir "$output/current_none/gate_proj" --category gate_proj --normalization current --protection none --steps 10000 > "$output/current_none_gate_proj.log" 2>&1
python -m experiments.cat_biaxial_scale.run --input-file "$input" --output-dir "$output/current_none/up_proj" --category up_proj --normalization current --protection none --steps 10000 > "$output/current_none_up_proj.log" 2>&1
python -m experiments.cat_biaxial_scale.run --input-file "$input" --output-dir "$output/current_none/down_proj" --category down_proj --normalization current --protection none --steps 10000 > "$output/current_none_down_proj.log" 2>&1
python -m experiments.cat_biaxial_scale.observe --prepared-dir /home/star/VAELLM/result/qwen_cat_biaxial_20261022/prepared --arm-dir "$output/current_none" --output-file "$output/current_none/observation.json" > "$output/current_none_observation.log" 2>&1

python -m experiments.cat_biaxial_scale.run --input-file "$input" --output-dir "$output/biaxial_none/gate_proj" --category gate_proj --normalization biaxial --protection none --steps 10000 > "$output/biaxial_none_gate_proj.log" 2>&1
python -m experiments.cat_biaxial_scale.run --input-file "$input" --output-dir "$output/biaxial_none/up_proj" --category up_proj --normalization biaxial --protection none --steps 10000 > "$output/biaxial_none_up_proj.log" 2>&1
python -m experiments.cat_biaxial_scale.run --input-file "$input" --output-dir "$output/biaxial_none/down_proj" --category down_proj --normalization biaxial --protection none --steps 10000 > "$output/biaxial_none_down_proj.log" 2>&1
python -m experiments.cat_biaxial_scale.observe --prepared-dir /home/star/VAELLM/result/qwen_cat_biaxial_20261022/prepared --arm-dir "$output/biaxial_none" --output-file "$output/biaxial_none/observation.json" > "$output/biaxial_none_observation.log" 2>&1

python -m experiments.cat_biaxial_scale.run --input-file "$input" --output-dir "$output/learned_none/gate_proj" --category gate_proj --normalization learned --protection none --steps 10000 > "$output/learned_none_gate_proj.log" 2>&1
python -m experiments.cat_biaxial_scale.run --input-file "$input" --output-dir "$output/learned_none/up_proj" --category up_proj --normalization learned --protection none --steps 10000 > "$output/learned_none_up_proj.log" 2>&1
python -m experiments.cat_biaxial_scale.run --input-file "$input" --output-dir "$output/learned_none/down_proj" --category down_proj --normalization learned --protection none --steps 10000 > "$output/learned_none_down_proj.log" 2>&1
python -m experiments.cat_biaxial_scale.observe --prepared-dir /home/star/VAELLM/result/qwen_cat_biaxial_20261022/prepared --arm-dir "$output/learned_none" --output-file "$output/learned_none/observation.json" > "$output/learned_none_observation.log" 2>&1

