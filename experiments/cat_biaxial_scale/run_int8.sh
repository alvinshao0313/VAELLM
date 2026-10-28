#!/usr/bin/env bash
set -euo pipefail
export OMP_NUM_THREADS=4
export CUBLAS_WORKSPACE_CONFIG=:4096:8
input=/home/star/VAELLM/result/qwen_cat_biaxial_20261022/prepared/inputs.pt
output=/home/star/VAELLM/result/qwen_cat_biaxial_20261022/formal
mkdir -p "$output"

python -m experiments.cat_biaxial_scale.run --input-file "$input" --output-dir "$output/current_int8/gate_proj" --category gate_proj --normalization current --protection int8 --steps 10000 > "$output/current_int8_gate_proj.log" 2>&1
python -m experiments.cat_biaxial_scale.run --input-file "$input" --output-dir "$output/current_int8/up_proj" --category up_proj --normalization current --protection int8 --steps 10000 > "$output/current_int8_up_proj.log" 2>&1
python -m experiments.cat_biaxial_scale.run --input-file "$input" --output-dir "$output/current_int8/down_proj" --category down_proj --normalization current --protection int8 --steps 10000 > "$output/current_int8_down_proj.log" 2>&1
python -m experiments.cat_biaxial_scale.observe --prepared-dir /home/star/VAELLM/result/qwen_cat_biaxial_20261022/prepared --arm-dir "$output/current_int8" --output-file "$output/current_int8/observation.json" > "$output/current_int8_observation.log" 2>&1

python -m experiments.cat_biaxial_scale.run --input-file "$input" --output-dir "$output/biaxial_int8/gate_proj" --category gate_proj --normalization biaxial --protection int8 --steps 10000 > "$output/biaxial_int8_gate_proj.log" 2>&1
python -m experiments.cat_biaxial_scale.run --input-file "$input" --output-dir "$output/biaxial_int8/up_proj" --category up_proj --normalization biaxial --protection int8 --steps 10000 > "$output/biaxial_int8_up_proj.log" 2>&1
python -m experiments.cat_biaxial_scale.run --input-file "$input" --output-dir "$output/biaxial_int8/down_proj" --category down_proj --normalization biaxial --protection int8 --steps 10000 > "$output/biaxial_int8_down_proj.log" 2>&1
python -m experiments.cat_biaxial_scale.observe --prepared-dir /home/star/VAELLM/result/qwen_cat_biaxial_20261022/prepared --arm-dir "$output/biaxial_int8" --output-file "$output/biaxial_int8/observation.json" > "$output/biaxial_int8_observation.log" 2>&1

python -m experiments.cat_biaxial_scale.run --input-file "$input" --output-dir "$output/learned_int8/gate_proj" --category gate_proj --normalization learned --protection int8 --steps 10000 > "$output/learned_int8_gate_proj.log" 2>&1
python -m experiments.cat_biaxial_scale.run --input-file "$input" --output-dir "$output/learned_int8/up_proj" --category up_proj --normalization learned --protection int8 --steps 10000 > "$output/learned_int8_up_proj.log" 2>&1
python -m experiments.cat_biaxial_scale.run --input-file "$input" --output-dir "$output/learned_int8/down_proj" --category down_proj --normalization learned --protection int8 --steps 10000 > "$output/learned_int8_down_proj.log" 2>&1
python -m experiments.cat_biaxial_scale.observe --prepared-dir /home/star/VAELLM/result/qwen_cat_biaxial_20261022/prepared --arm-dir "$output/learned_int8" --output-file "$output/learned_int8/observation.json" > "$output/learned_int8_observation.log" 2>&1

